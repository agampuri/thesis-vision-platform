"""Rule-based intent router: text command -> Intent. Deterministic and testable (E13)."""
import re
from dataclasses import dataclass, field


@dataclass
class Intent:
    action: str                 # pick | pick_place | sort | query_count | query_find | chess | unknown
    object_query: str = ""
    target_zone: str = ""
    deictic: bool = False       # "this"/"that"/"it" -> resolve via pointing
    confidence: float = 0.0
    raw: str = ""
    params: dict = field(default_factory=dict)


_ARTICLES = re.compile(r"\b(the|a|an|please|my)\b")
_DEICTIC = re.compile(r"^(this( one)?|that( one)?|it)$")


def _clean(s):
    s = _ARTICLES.sub(" ", s.lower())
    return re.sub(r"\s+", " ", s).strip(" .!?,")


class IntentRouter:
    def __init__(self, zones_cfg=None, objects_cfg=None, logger=None):
        self.logger = logger
        self.zones = {}
        for name, z in ((zones_cfg or {}).get('zones') or {}).items():
            self.zones[name] = [name.replace('_', ' ')] + list(z.get('aliases', []))
        self.objects = (objects_cfg or {}).get('objects') or {}

    def _match_zone(self, phrase):
        p = _clean(phrase)
        for name, aliases in self.zones.items():
            for a in aliases:
                if a in p:
                    return name
        return ""

    def resolve_object(self, phrase):
        """-> (detector_query, catalogue_entry_or_None)."""
        p = _clean(phrase)
        for name, o in self.objects.items():
            for a in [name.replace('_', ' ')] + list(o.get('aliases', [])):
                if a in p or p in a:
                    return o.get('query', a), o
        return p, None

    def route(self, text):
        t = _clean(text)
        if not t:
            return Intent('unknown', raw=text)
        if t.startswith('chess') or 'play chess' in t:
            return Intent('chess', confidence=0.95, raw=text,
                          params={'args': t.replace('play chess', '').replace('chess', '').strip()})
        if re.search(r"\b(sort|tidy( up)?|clear|clean( up)?)\b", t):
            return Intent('sort', confidence=0.9, raw=text)
        m = re.search(r"\bhow many (.+?)( are .*)?$", t) or re.search(r"\bcount( the)? (.+)$", t)
        if m:
            obj = m.group(1) if 'how many' in t else m.group(2)
            return Intent('query_count', object_query=_clean(obj), confidence=0.9, raw=text)
        m = re.search(r"\b(where is|find|locate|show me) (.+)$", t)
        if m:
            return Intent('query_find', object_query=_clean(m.group(2)), confidence=0.9, raw=text)
        m = re.search(r"\b(put|place|move|drop) (.+?) (?:in|into|on|onto|to) (.+)$", t)
        if m:
            obj, tgt = _clean(m.group(2)), m.group(3)
            return Intent('pick_place', object_query=obj, target_zone=self._match_zone(tgt),
                          deictic=bool(_DEICTIC.match(obj)), confidence=0.9, raw=text,
                          params={'target_raw': _clean(tgt)})
        m = re.search(r"\b(pick up|pick|grab|take|get) (.+)$", t)
        if m:
            obj = _clean(m.group(2))
            return Intent('pick', object_query=obj, deictic=bool(_DEICTIC.match(obj)),
                          confidence=0.85, raw=text)
        return Intent('unknown', confidence=0.1, raw=text)
