"""공식 평가 코드(rouge_metric.py)를 수정 없이 불러오기 위한 호환 모듈.

konlpy.tag.Mecab은 시스템에 mecab-ko와 JDK가 설치되어 있어야 한다.
python-mecab-ko는 같은 사전(mecab-ko-dic)을 wheel로 제공하고 .morphs() 인터페이스가 같으므로 이를 대신 쓴다.
"""
from mecab import MeCab as Mecab  # noqa: F401
