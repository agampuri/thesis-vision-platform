#!/usr/bin/env python3
"""
Multi-skill platform entry point. Chess stays available standalone via main.py.

  Terminal 1:  ros2 launch xarm_moveit_config lite6_moveit_realmove.launch.py \
                   robot_ip:=192.168.1.175 add_gripper:=true     (or the fake launch)
  Terminal 2:  cd ~/chess_remote/robot/src && python3 platform_main.py

Commands at the prompt:
  put the red cube in the blue bin     pick up the marker
  sort the table / tidy up             how many red cubes are there
  where is the glue stick              chess --color white --mode ai --vision
  status | help | quit
"""
import argparse
import asyncio
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml


def load_cfg(name):
    paths = [os.path.expanduser(f"~/chess_remote/config/{name}"),
             os.path.join(os.path.dirname(__file__), '..', '..', 'config', name)]
    for p in paths:
        p = os.path.abspath(p)
        if os.path.exists(p):
            with open(p) as f:
                return yaml.safe_load(f) or {}
    raise FileNotFoundError(name)


def build_world(args, logger):
    """rclpy + node + services. Returned as (node, services, manager)."""
    import rclpy
    from chess_robot.nodes.chess_node import ChessNode
    from chess_robot.performance_logger import PerformanceLogger
    from chess_robot.platform.intent_router import IntentRouter
    from chess_robot.platform.motion_service import MotionService
    from chess_robot.platform.safety_monitor import SafetyMonitor
    from chess_robot.platform.grasp_planner import GraspPlanner
    from chess_robot.platform.services import Services
    from chess_robot.platform.skill_manager import SkillManager
    from chess_robot.skills.pick_place_skill import PickPlaceSkill
    from chess_robot.skills.sort_skill import SortSkill
    from chess_robot.skills.query_skill import QuerySkill

    rclpy.init()
    try:
        perf = PerformanceLogger(logger)
    except TypeError:
        perf = PerformanceLogger()
    node = ChessNode(perf_logger=perf)

    zones = load_cfg('zones.yaml')
    objects = load_cfg('objects.yaml')

    vision = None
    if not args.no_vision:
        from chess_robot.vision.vision_system import VisionSystem
        vcfg = node.movement.planner.config.get('vision')
        if not vcfg:
            logger.error("No 'vision:' block in board_config.yaml")
            sys.exit(1)
        vision = VisionSystem(vcfg, logger=logger)
        vision.start()
        ok = vision.update()
        logger.info(f"vision: initial update {'OK' if ok else 'FAILED'} "
                    f"health={vision.health()}")
        node.movement.planner.set_vision(vision)

    motion = MotionService(node, zones, logger=logger)
    safety = SafetyMonitor(zones, logger=logger)
    router = IntentRouter(zones, objects, logger=logger)
    grasp = None
    open_vocab = None
    if vision is not None:
        grasp = GraspPlanner(vision.extrinsics, zones, logger=logger)
        from chess_robot.vision.open_vocab import OpenVocabDetector
        open_vocab = OpenVocabDetector(conf=0.25, logger=logger)

    services = Services(node=node, vision=vision, motion=motion, grasp=grasp,
                        safety=safety, router=router, open_vocab=open_vocab,
                        zones=zones, objects=objects, logger=logger)
    manager = SkillManager(logger=logger)
    for skill in (PickPlaceSkill(), SortSkill(), QuerySkill()):
        manager.register(skill)
    return node, services, manager


def teardown_world(node, services):
    import rclpy
    try:
        if services.vision:
            services.vision.stop()
    except Exception:
        pass
    try:
        node.destroy_node()
    except Exception:
        pass
    try:
        rclpy.shutdown()
    except Exception:
        pass


def chess_handoff(intent, node, services, logger):
    """Release ROS, run the proven standalone chess app, re-acquire afterwards."""
    from chess_robot.skills.chess_skill import build_command
    cmd = build_command(intent)
    logger.info(f"Chess handoff: {' '.join(cmd[1:])}")
    teardown_world(node, services)
    try:
        subprocess.call(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    finally:
        logger.info("Chess finished — restarting platform services...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-vision', action='store_true',
                    help='start without the camera (chess handoff only)')
    args = ap.parse_args()

    from chess_robot.logging_utils import setup_logging
    logger = setup_logging('platform')

    node, services, manager = build_world(args, logger)
    print(__doc__)
    try:
        while True:
            try:
                text = input("\nplatform » ").strip()
            except EOFError:
                break
            if not text:
                continue
            low = text.lower()
            if low in ('quit', 'exit', 'q'):
                break
            if low == 'help':
                print(__doc__)
                continue
            if low == 'status':
                h = services.vision.health() if services.vision else 'vision off'
                print(f"  vision: {h}\n  gripper backend: {services.motion.backend}")
                continue
            intent = services.router.route(text)
            if intent.action == 'chess':
                chess_handoff(intent, node, services, logger)
                node, services, manager = build_world(args, logger)
                continue
            if intent.action == 'unknown':
                print("  Didn't understand. Try: 'put the red cube in the blue bin', "
                      "'sort the table', 'how many markers', 'chess'.")
                continue
            if services.vision is None:
                print("  Vision is off (--no-vision); only 'chess' is available.")
                continue
            skill = manager.select(intent)
            if skill is None:
                print(f"  No skill can handle: {intent.action}")
                continue
            print(f"  -> {skill.name} ({intent.action}: '{intent.object_query}'"
                  f"{' -> ' + intent.target_zone if intent.target_zone else ''})")
            asyncio.new_event_loop().run_until_complete(
                manager.run(skill, intent, services))
    finally:
        teardown_world(node, services)
        print("Platform stopped.")


if __name__ == '__main__':
    main()
