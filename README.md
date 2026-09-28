# 한국문화 멀티모달 질의응답에서 지식 활용과 답변 최적화의 효과 분석

국립국어원 인공지능(AI)말평 [한국문화 멀티모달 질의응답 과제](https://kli.korean.go.kr/m/taskOrdtm/taskList.do?taskOrdtmId=211)

2026년도 한글 및 한국어 정보처리 학술대회(HCLT 2026): 
한국문화 멀티모달 질의응답에서 지식 활용과 답변 최적화의 효과 분석

한국외국어대학교 HCNLP 연구실
배준영, 강어진, 조수민, 김주애

시각 언어 모델(VLM)이 사진을 관찰문으로 옮기고 한국어 언어 모델(LLM)이 답하는 릴레이 구조 위에, 문항 형식별 LoRA 어댑터.

1. 참고 자료를 문맥으로 주는 RAG와 검색 조건부 미세조정(RAFT)
2. 단답형·서술형 직접 선호 최적화(DPO)
3. 서술형 최소 베이즈 위험(MBR) 후보 선택
4. 이미지 검색 유사도 임곗값에 따른 참고 자료의 효용

## 결과

| 구성 | 선다형 | 단답형 | 서술형 |
|---|---|---|---|
| VLM 직접 답 | 78.31 | 32.04 | 34.66 |
| 최종 시스템 | 89.16 | 66.02 | 41.80 |

## 구조

```
src/          추론
  common.py             프롬프트, 답 정규화, 출력 규격 검사, 입출력
  prep_data.py          EXIF 방향 보정과 이미지 축소
  observe.py            릴레이 1단계: VLM 관찰문 (3.1절)
  cropzoom.py           불확실한 글자 영역의 확대 판독
  answer_vlm.py         VLM 직접 답 (선다형)
  answer_text.py        릴레이 2단계: LLM 답. SFT, RAG, RAFT 공용
  mc_vote.py            선다형 투표와 다중 선택 재투표
  sa_postprocess.py     단답형 음절·어절 재시도와 후보 필터
  sample_candidates.py  greedy와 확률적 생성 후보
  mbr_select.py         서술형 MBR 후보 선택 (3.4절)
  score.py              공식 평가 지표
  make_submission.py    제출 파일 생성
train/        학습
  build_sft_data.py     문항 형식별 SFT 데이터 (3.2절)
  train_lora_text.py    LLM LoRA 학습. SFT와 RAFT 공용
  train_lora_vlm.py     VLM LoRA 학습. 선다형 어댑터
  build_raft_data.py    참고 자료를 붙인 RAFT 데이터 (3.3절)
  build_dpo_pairs.py    문항 형식별 DPO 선호 쌍 (3.5절)
  train_dpo.py          DPO 학습
retrieval/    검색
  image_index.py        DINOv2 이미지 색인과 검색
  text_index.py         BM25 + BGE-M3 글 색인과 RRF 검색
  build_gate.py         선택적 검색 게이트와 참고 자료 구성
official/     국립국어원 공식 평가 코드 (teddysum/korean_evaluation의 rouge_metric.py)
scripts/      vLLM 서버 실행 스크립트
configs/      게이트 용어 목록 예시
```

모든 하이퍼파라미터는 각 스크립트의 `argparse` 인자로 관리. 기본값은 논문의 설정입니다. `python <스크립트> --help`로 확인.

## 설치

```bash
pip install -r requirements.txt
```

추론은 H200(143 GB) 한 장에서 수행. 한 번에 한 모델만 구동하므로 VLM 단계와 LLM 단계를 차례로 실행.

## 데이터와 모델

데이터는 저장소에 담지 않습니다. 아래에서 받아 주세요.

| 자료 | 받는 곳 | 용도 |
|---|---|---|
| 대회 데이터(문항 JSON, 이미지) | [국립국어원 인공지능(AI)말평 과제 페이지](https://kli.korean.go.kr/benchmark/taskOrdtm/taskList.do?taskOrdtmId=211) | 학습, 검증, 평가 |
| e뮤지엄 | https://www.emuseum.go.kr/ | 이미지 색인 |
| 국가유산포털 | https://www.heritage.go.kr/ | 이미지 색인 |
| 포토코리아 | https://phoko.visitkorea.or.kr/ | 이미지 색인 |
| 우리말샘 | https://opendict.korean.go.kr/ | 글 근거 |
| 한국민족문화대백과사전 | https://encykorea.aks.ac.kr/ | 글 근거 |

| 모델 | 받는 곳 | 역할 |
|---|---|---|
| Qwen3.8-27B | https://huggingface.co/Qwen/Qwen3.8-27B | 시각 관찰, 선다형 직접 답 |
| A.X-4.0 | https://huggingface.co/skt/A.X-4.0 | 한국어 답변 |
| DINOv2 | https://huggingface.co/facebook/dinov2-large | 이미지 검색 |
| BGE-M3 | https://huggingface.co/BAAI/bge-m3 | 글 검색 |

외부 자료는 아래 형식으로 정리해 두면 검색 스크립트가 읽습니다.

- 이미지: 한 폴더에 모은 이미지 파일과 `manifest.json` (`{파일 이름: {"name": 표제어, "desc": 설명, "source": 출처}}`)
- 글: `articles.jsonl` (한 줄에 `{"id", "headword", "definition", "body", "source"}`)

## 실행 순서

아래에서 `data/`는 대회 데이터를 푼 폴더, `ext/`는 외부 자료, `work/`는 중간 산출물, `adapters/`는 학습한 어댑터입니다.

### 1. 데이터 준비

```bash
for s in train validation test; do
  python src/prep_data.py --data-root <zip 폴더> --work data --split $s --max-sides 1280,1792
done
```

### 2. 관찰문 생성 (VLM 서버)

```bash
bash scripts/serve_vlm.sh adapters 8000
for s in train test; do
  python src/observe.py --data data/$s.json --images data/images/${s}_1792 --out work/obs_$s.jsonl
  python src/cropzoom.py --data data/$s.json --obs work/obs_$s.jsonl \
      --cache data/images/${s}_1792 --orig data/images/$s --out work/obs_${s}_zoom.jsonl
done
```

### 3. 문항 형식별 SFT

```bash
python train/build_sft_data.py --data data/train.json --obs work/obs_train_zoom.jsonl --out-dir work/sft

python train/train_lora_vlm.py  --sft work/sft/vlm_mc.jsonl --images data/images/train_1280 --out adapters/a-mc
python train/train_lora_text.py --sft work/sft/text_mc.jsonl --out adapters/a-mc-text --lr 5e-5 --load-8bit
python train/train_lora_text.py --sft work/sft/text_sa.jsonl --out adapters/a-sa      --lr 5e-5 --load-8bit
python train/train_lora_text.py --sft work/sft/text_la.jsonl --out adapters/a-la      --lr 1e-4 --load-8bit
```

| 어댑터 | 기반 모델 | 학습 데이터 | 에폭 | 학습률 |
|---|---|---|---|---|
| A-MC | Qwen3.8-27B | 문제 풀이·정답 서술·이미지 설명 3,000행 | 2 | 1e-4 |
| A-MC′ | A.X-4.0 | 선다형 문제 풀이 + 정답 서술 1,518행 | 3 | 5e-5 |
| A-SA | A.X-4.0 | 단답형 254문항 (관찰문 + 질문 → 정답) | 3 | 5e-5 |
| A-LA | A.X-4.0 | 서술형 228문항 (질문 → 정답, 관찰문 없음) | 3 | 1e-4 |

LoRA는 모두 r = 16, α = 32, 드롭아웃 0.05. VLM은 시각 인코더를 고정.

### 4. 선택적 검색과 RAFT

```bash
python retrieval/image_index.py build --image-dir ext/images --index-dir work/img_index
python retrieval/text_index.py  build --corpus ext/articles.jsonl --index-dir work/txt_index

for s in train test; do
  python retrieval/image_index.py query --index-dir work/img_index --manifest ext/manifest.json \
      --data data/$s.json --images data/images/${s}_1280 --out work/img_hits_$s.jsonl
  python retrieval/build_gate.py --data data/$s.json --obs work/obs_${s}_zoom.jsonl \
      --image-hits work/img_hits_$s.jsonl --index-dir work/txt_index \
      --keywords configs/gate_keywords.example.json --sim-threshold 0.88 --out work/gate_$s.json
  python train/build_raft_data.py --split $s --data data/$s.json --obs work/obs_${s}_zoom.jsonl \
      --gate work/gate_$s.json --out-dir work/raft
done

# SFT 어댑터를 초깃값으로 3에폭, 학습률 1e-4, 최대 길이 6,144토큰
python train/train_lora_text.py --sft work/raft/text_train_sa.jsonl --init-adapter adapters/a-sa \
    --out adapters/raft-sa --max-len 6144 --bs 1 --grad-accum 4 --load-8bit
python train/train_lora_text.py --sft work/raft/text_train_la.jsonl --init-adapter adapters/a-la \
    --out adapters/raft-la --max-len 6144 --bs 1 --grad-accum 4 --load-8bit
python train/train_lora_vlm.py --sft work/raft/vlm_train_mc.jsonl --images data/images/train_1280 \
    --init-adapter adapters/a-mc --out adapters/raft-mc --epochs 3 --grad-accum 4 --max-len 6144
```

게이트는 유물 관련 용어의 키워드 조건 또는 이미지 최고 유사도 s > 0.88. `configs/gate_keywords.example.json`은 용어 목록의 형식을 보여 주는 예시. 유사도 임곗값 실험(5.3절)은 `build_gate.py --no-keyword --sim-threshold <값>`으로 적용 범위를 바꿔 실행.

### 5. 서술형 DPO

```bash
bash scripts/serve_llm.sh adapters 8000

# 학습 분할에서 후보를 생성하고 정답과 비교해 선호 쌍을 만든다
python src/sample_candidates.py --data data/train.json --obs work/obs_train_zoom.jsonl --forms LA \
    --model a-la --n 16 --out work/la_cands_train.jsonl
python train/build_dpo_pairs.py la --data data/train.json --obs work/obs_train_zoom.jsonl \
    --cands work/la_cands_train.jsonl --chosen best --min-gap 15 --out work/dpo/pairs_la.jsonl

# 서버를 내린 뒤 학습한다
python train/train_dpo.py --form LA --pairs work/dpo/pairs_la.jsonl --adapter adapters/a-la \
    --out adapters/dpo-la --beta 0.5 --epochs 1 --lr 5e-6 --grad-accum 8 --max-len 2048
```

단답형 DPO(표 3)는 `build_dpo_pairs.py sa`와 `train_dpo.py --form SA`로 실행.

### 6. 추론

```bash
# (VLM 서버) 선다형: 선다형 어댑터, 원본 VLM, 게이트 문항의 RAFT 답
python src/answer_vlm.py --data data/test.json --images data/images/test_1280 --model a-mc  --out work/mc_sft.jsonl
python src/answer_vlm.py --data data/test.json --images data/images/test_1280 --model bench --out work/mc_base.jsonl
python src/answer_vlm.py --data data/test.json --images data/images/test_1280 --model raft-mc \
    --rows work/raft/rows_test.jsonl --out work/mc_raft.jsonl

# (LLM 서버) 선다형 릴레이 답과 다중 선택 텍스트 어댑터 답
python src/answer_text.py --data data/test.json --obs work/obs_test_zoom.jsonl --forms MC --model bench     --out work/mc_relay.jsonl
python src/answer_text.py --data data/test.json --obs work/obs_test_zoom.jsonl --forms MC --model a-mc-text --out work/mc_text.jsonl
python src/mc_vote.py --data data/test.json --sft work/mc_sft.jsonl --base work/mc_base.jsonl \
    --relay work/mc_relay.jsonl --multi work/mc_text.jsonl --raft work/mc_raft.jsonl --out work/mc_final.jsonl

# 단답형: SFT 답과 게이트 문항의 RAFT 답에 출력 규격 교정을 적용한다
python src/answer_text.py --data data/test.json --obs work/obs_test_zoom.jsonl --forms SA --model a-sa --out work/sa.jsonl
python src/answer_text.py --data data/test.json --obs work/obs_test_zoom.jsonl --forms SA --model raft-sa \
    --rows work/raft/rows_test.jsonl --out work/sa_raft.jsonl
python src/sa_postprocess.py --data data/test.json --obs work/obs_test_zoom.jsonl --preds work/sa.jsonl \
    --cands work/sa_raft.jsonl --model a-sa --out work/sa_final.jsonl
python src/sa_postprocess.py --data data/test.json --obs work/obs_test_zoom.jsonl --preds work/sa_raft.jsonl \
    --cands work/sa.jsonl --model a-sa --out work/sa_raft_final.jsonl

# 서술형: greedy 1개와 확률적 생성 31개(temperature 0.7, 누적 확률 상한 0.95)에서 MBR로 고른다
python src/sample_candidates.py --data data/test.json --obs work/obs_test_zoom.jsonl --forms LA \
    --model dpo-la --n 31 --out work/la_cands_test.jsonl
python src/mbr_select.py --cands work/la_cands_test.jsonl --n 32 --out work/la_mbr.jsonl
python src/answer_text.py --data data/test.json --obs work/obs_test_zoom.jsonl --forms LA --model raft-la \
    --rows work/raft/rows_test.jsonl --out work/la_raft.jsonl

# 제출 파일: 뒤에 적은 게이트 문항의 답이 앞의 답을 덮어쓴다
python src/make_submission.py --data data/test.json \
    --preds work/mc_final.jsonl work/sa_final.jsonl work/la_mbr.jsonl work/sa_raft_final.jsonl work/la_raft.jsonl \
    --out submission.json
```

RAG 비교 구성(표 2)은 같은 참고 자료와 예제 두 개를 주고 추가 학습 없이 답.

```bash
python src/answer_text.py --data data/test.json --obs work/obs_test_zoom.jsonl --forms SA,LA --model bench \
    --rows work/raft/rows_test.jsonl --shots work/raft/icl.json --out work/rag.jsonl
```

### 7. 채점

평가 데이터의 정답은 공개되지 않으므로 검증 데이터로 채점.

```bash
python src/score.py --gold data/validation.json --pred work/preds_validation.jsonl
```

## 주요 하이퍼파라미터

| 항목 | 값 | 인자 |
|---|---|---|
| LoRA | r 16, α 32, 드롭아웃 0.05 | `--r --alpha --dropout` |
| 관찰 해상도 | 긴 변 1,792픽셀 | `prep_data.py --max-sides` |
| 게이트 유사도 임곗값 | 0.88 | `build_gate.py --sim-threshold` |
| 선택 문서 조각 수 | 2 | `build_gate.py --top-chunks` |
| RAFT | 3에폭, 학습률 1e-4, 최대 6,144토큰 | `--epochs --lr --max-len` |
| MBR 후보 수 | 32 | `mbr_select.py --n` |
| 후보 생성 | temperature 0.7, 누적 확률 상한 0.95 | `sample_candidates.py --temperature --top-p` |
| DPO | β 0.5, 1에폭, 학습률 5e-6, 유효 배치 8 | `train_dpo.py --beta --epochs --lr --grad-accum` |
| 서술형 선호 쌍 점수 차이 | 15점 이상 | `build_dpo_pairs.py --min-gap` |

## 참고

- 대회 규정에 따라 추론에는 외부 API 모델을 쓰지 않습니다. 검색도 모두 로컬에서 실행합니다.
- 평가 문항은 프롬프트나 예제에 넣지 않습니다. 학습 데이터와 문맥 내 예제는 학습 분할에서만 만듭니다.
- `official/rouge_metric.py`는 국립국어원 공식 평가 코드([teddysum/korean_evaluation](https://github.com/teddysum/korean_evaluation))에서 가져왔습니다.

## 감사의 글

본 연구는 문화체육관광부 및 한국콘텐츠진흥원의 2026년도 문화기술 연구개발 사업으로 수행되었음
(과제명: 개인 맞춤형 국어 생활 종합 상담 서비스를 위한 한국어 지식 연계 AI 에이전트 개발, 과제번호: RS-2026-25506607).
