from dataclasses import dataclass, field
from typing import Optional
from datetime import timedelta

def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))


@dataclass
class TimedText:
    start: Optional[float] = 0
    end: Optional[float] = 0
    text: Optional[str] = ''
    speaker: Optional[int] = -1
    probability: Optional[float] = None
    is_dummy: Optional[bool] = False

@dataclass
class ASRToken(TimedText):
    def with_offset(self, offset: float) -> "ASRToken":
        """Return a new token with the time offset added."""
        return ASRToken(self.start + offset, self.end + offset, self.text, self.speaker, self.probability)

@dataclass
class Sentence(TimedText):
    pass

@dataclass
class Transcript(TimedText):
    pass

@dataclass
class SpeakerSegment(TimedText):
    """Represents a segment of audio attributed to a specific speaker.
    No text nor probability is associated with this segment.
    """
    pass

@dataclass
class Translation(TimedText):
    pass

@dataclass
class Silence():
    duration: float


@dataclass
class Line(TimedText):
    translation: str = ''

    def to_dict(self):
        return {
            'speaker': int(self.speaker),
            'text': self.text,
            'translation': self.translation,
            'start': format_time(self.start),
            'end': format_time(self.end),
        }

@dataclass
class FrontData():
    status: str = ''
    error: str = ''
    lines: list[Line] = field(default_factory=list)
    buffer_transcription: str = ''
    buffer_diarization: str = ''
    remaining_time_transcription: float = 0.
    remaining_time_diarization: float = 0.

    def to_dict(self):
        _dict = {
            'status': self.status,
            'lines': [line.to_dict() for line in self.lines],
            'buffer_transcription': self.buffer_transcription,
            'buffer_diarization': self.buffer_diarization,
            'remaining_time_transcription': self.remaining_time_transcription,
            'remaining_time_diarization': self.remaining_time_diarization,
        }
        if self.error:
            _dict['error'] = self.error
        return _dict

@dataclass
class State():
    tokens: list = field(default_factory=list)
    translated_segments: list = field(default_factory=list)
    buffer_transcription: str = ""
    buffer_diarization: str = ""
    end_buffer: float = 0.0
    end_attributed_speaker: float = 0.0
    remaining_time_transcription: float = 0.0
    remaining_time_diarization: float = 0.0

@dataclass
class FormattedResponse():
    """包装format_output返回的tuple，提供to_dict方法"""
    lines: list = field(default_factory=list)
    undiarized_text: list = field(default_factory=list)
    buffer_transcription: str = ""
    buffer_diarization: str = ""

    @classmethod
    def from_tuple(cls, response_tuple):
        """从format_output返回的tuple创建对象"""
        if isinstance(response_tuple, tuple) and len(response_tuple) >= 4:
            lines, undiarized_text, buffer_transcription, buffer_diarization = response_tuple[:4]
            return cls(lines, undiarized_text, buffer_transcription, buffer_diarization)
        return cls()

    def to_dict(self):
        """转换为字典格式，用于WebSocket传输"""
        lines_dict = []
        for line in self.lines:
            if hasattr(line, 'to_dict'):
                lines_dict.append(line.to_dict())
            else:
                # 兼容处理
                lines_dict.append({
                    'speaker': getattr(line, 'speaker', -1),
                    'text': getattr(line, 'text', ''),
                    'start': getattr(line, 'start', 0),
                    'end': getattr(line, 'end', 0),
                })

        return {
            'lines': lines_dict,
            'undiarized_text': self.undiarized_text,
            'buffer_transcription': self.buffer_transcription,
            'buffer_diarization': self.buffer_diarization,
        }