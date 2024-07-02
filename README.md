## 감정 분석 일기장 
- 프로젝트는 5명이 진행하였습니다.
- 컨텐츠에 개인의 감정에 따른 이미지 생성을 위해 stable diffusion을 이용한 생성형 이미지 제공을 계획하였으나, 
- 프로젝트의 효율성과 stable diffusion의 안정성 문제로 제외하였습니다.
- 모든 부분 작업에 참여했습니다.

### 프로젝트 배경
- 국내 우울증 환자가 100만명을 넘었고 정신질환 환자는 465만명을 넘겼다.
- 이런 환자들의 추세는 특정세대만이 아닌 모든 세대에서 치료 건수가 증가하고 있다.
- 자신의 상태를 인지하지 못하여 골든타임을 놓치는 경우가 많이 발생함.
- 고로 예방에 초점을 맞춘 서비스로 자신의 감정을 스스로 파악하는 데 도움을 주어 스스로를 바라볼 수 있는 시간을 제공.

### 프로젝트 수행절차
1. 감정분석을 위해 문장에 감정이 레이블링 된 형태의 데이터 확보
2. 데이터 전처리 (감정의 분류 및 통합 등)
3. LSTM 자체모델과 KoBert모델 구축 및 비교 평가
4. 서비스 구상( 감정의 레이더 차트, 감정 기반 크롤링을 통해 음악이나 영상 추천 제공)
5. 웹 구현 > 웹뷰를 사용한 웹/앱 포팅

### 데이터 수집 및 전처리
- AI hub에서 감성 대화 말뭉치 데이터셋을 사용
- 약 4만건의 데이터로 구성 (대감정 기쁨, 당황,분노, 불안, 상처, 슬픔 6개의 감정)
- 이상치와 결측치는 없었으며, 대분류 감정과 사람문장1, 2만을 사용

### 모델 평가
![image](https://github.com/msdlml/Project-ai-diary-/assets/156978979/c8f0006e-1b30-4261-bc3d-a35c2b42bafa)

-KoBert 파인튜닝 모델

![image](https://github.com/msdlml/Project-ai-diary-/assets/156978979/6b7dde80-63f5-42d7-b339-b30ad8a8a95f)

-정확도 뿐 아니라 실험적인 문장(오타, 다중감정 등)으로 모델을 테스트 한 결과 적절히 분류
![image](https://github.com/msdlml/Project-ai-diary-/assets/156978979/82013cd5-20ab-414b-84aa-45b3a0eea4a6)

-KoBert 모델 정확도 개선 부분
![image](https://github.com/msdlml/Project-ai-diary-/assets/156978979/6c30f95d-7c35-49b8-9512-96bfdc20b186)

### 서비스 개발 방식과 구현
![image](https://github.com/msdlml/Project-ai-diary-/assets/156978979/6864d766-510b-4382-be79-d4b48d0efd07)

![image](https://github.com/msdlml/Project-ai-diary-/assets/156978979/53404381-5fb3-46ab-9d24-bcc34a23c593)

### 구현 화면
![image](https://github.com/msdlml/Project-ai-diary-/assets/156978979/c3b75b66-0a93-4625-947a-c25acf8fbdbb)

![image](https://github.com/msdlml/Project-ai-diary-/assets/156978979/775dacc6-3402-4df8-8004-ce8d75590eb4)

![image](https://github.com/msdlml/Project-ai-diary-/assets/156978979/ef1a5054-b4f3-4011-8753-5147759c66fb)

![image](https://github.com/msdlml/Project-ai-diary-/assets/156978979/1d501acc-e73f-4c89-ac74-047ff23e9b4f)

![image](https://github.com/msdlml/Project-ai-diary-/assets/156978979/74dc9a6b-dd79-42d4-b029-a88b78250bc1)

- 웹뷰는 코틀린을 활용하여 구현하였습니다.
