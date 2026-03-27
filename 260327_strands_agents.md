# AWS Strands Agents SDK 소개 및 주요 프레임워크 비교

Amazon Neptune을 활용한 Graph RAG 애플리케이션 구축 시, 아키텍처에 **Strands Agents SDK**가 있어서 소개해드리려고 합니다.

최근 AWS에서 오픈소스로 공개한 **Strands Agents SDK**는 기존 프레임워크들의 복잡성을 덜어내고, LLM 자체의 추론 능력에 기반한 가볍고 강력한 대안으로 주목받고 있습니다.

## 1. Strands Agents SDK 간략 소개

[Strands Agents SDK](https://strandsagents.com/)는 복잡한 워크플로우를 하드코딩하는 대신, **모델 중심(Model-driven) 접근 방식**을 채택하여 단 몇 줄의 코드만으로 프로덕션 레벨의 AI 에이전트를 구축할 수 있는 오픈소스 프레임워크입니다.

### 🌟 핵심 특징

1. **모델 중심 오케스트레이션 (Model-driven):** 개발자가 노드(Node)와 엣지(Edge)를 일일이 그리는 대신, LLM이 스스로 계획(Plan)하고, 생각을 연결(Chain of thought)하며, 도구를 선택하도록 위임합니다.

2. **압도적인 코드 간결성:** `Model`, `Prompt`, `Tools` 세 가지 요소만 선언하면 에이전트 루프가 자동으로 동작합니다. Python 함수 위에 `@tool` 데코레이터만 붙이면 즉시 LLM이 사용할 수 있는 도구로 변환됩니다.

3. **MCP(Model Context Protocol) 기본 지원:** 수천 개의 사전 구축된 MCP 서버(Notion, Confluence, Graph DB 등)와 즉시 연결할 수 있어, Neptune 데이터베이스 쿼리 도구를 쉽게 확장할 수 있습니다.

4. **유연한 생태계 (Model Agnostic):** Amazon Bedrock 뿐만 아니라 OpenAI, Anthropic, 로컬 모델(Ollama) 등 대부분의 LLM을 자유롭게 교체하며 사용할 수 있습니다.

---
## 2. 단순 SDK 사용성 관점에서의 비교 (Developer Experience)

목적과 설계 철학을 깊게 파고들기 전에, 순수한 개발 라이브러리(SDK) 관점만 놓고 보면 Strands Agents는 기존 인기 프레임워크들의 장점을 훌륭하게 흡수하여 직관적인 개발 경험을 제공합니다.

* **LangChain과의 유사성 (간편한 도구 결합):** LangChain이 `@tool` 데코레이터나 `bind_tools()`를 통해 일반 파이썬 함수를 LLM 도구로 쉽게 변환하듯, Strands Agents 역시 복잡한 클래스 상속이나 추상화 없이 가장 파이썬다운(Pythonic) 방식으로 Agent와 Tool을 즉각적으로 연결합니다.

* **LiteLLM과의 유사성 (통합 모델 인터페이스):** LiteLLM이 수많은 LLM 벤더의 복잡한 API를 하나의 표준 형식으로 묶어주듯, Strands Agents도 단일 인터페이스로 여러 모델을 호출합니다. 각 벤더의 API 규격을 따로 학습할 필요 없이, 코드 한 줄 변경(`model="anthropic.claude-3"` -> `model="amazon.titan"`)만으로 모델 스위칭이 가능합니다.

---
## 3. 목적 및 설계 철학 중심의 주요 프레임워크 비교

단순한 기능 호출을 넘어 애플리케이션의 **오케스트레이션 아키텍처**를 구성할 때, Strands Agents는 시중의 유명 SDK들과 비교하여 뚜렷한 포지셔닝을 가집니다.

| 비교 항목 | Strands Agents (AWS) | LangChain / LangGraph | ADK (CrewAI, AutoGen 등) | LiteLLM | 
| ----- | ----- | ----- | ----- | ----- | 
| **설계 철학** | **Model-driven** (모델의 자율적 추론 우선) | **Graph/Workflow-driven** (명시적 상태 머신 통제) | **Role/Conversation-driven** (에이전트 간 대화와 역할극) | **I/O Standardization** (API 프록시 및 표준화) | 
| **오케스트레이션  방식** | LLM이 프롬프트와 도구 목록을 보고 스스로 실행 루프(Loop) 결정 | 개발자가 State, Node, Conditional Edge를 코드로 직접 설계 | 에이전트들에게 페르소나를 부여하고 순차적/계층적 대화로 작업 수행 | (에이전트 프레임워크가 아님) 100+개 LLM의 API 호출 형식을 OpenAI 규격으로 통일 | 
| **학습 곡선 및  코드 복잡도** | **매우 낮음** (단 5줄 내외의 코드로 기본 에이전트 구동 가능) | **매우 높음** (개념이 방대하고 상태 관리 구조가 복잡함) | **보통** (태스크와 에이전트 정의 위주로 직관적임) | **낮음** (단순 라우팅 및 프록시 설정용) | 
| **도구(Tool)  통합 방식** | `@tool` 데코레이터 및 **MCP(Model Context Protocol)** 네이티브 지원 | 자체 `Tool` 클래스 래핑 및 방대한 커뮤니티 통합(Integrations) 패키지 | LangChain 도구 호환 또는 자체 커스텀 함수 사용 | 도구 통합 기능보다는 함수 호출(Function Calling) 파싱에 집중 | 
| **Graph RAG /  AWS 연동성** | **최상** (Bedrock, S3 Vector, Lambda 등과 네이티브 통합, MCP로 Neptune 연동 용이) | 우수 (다양한 Graph DB 라이브러리가 존재하나 구조가 무거워질 수 있음) | 보통 (Multi-Agent로 Cypher 검증 루프 설계는 좋으나 구현이 까다로움) | 해당 없음 (보통 Strands나 LangChain 내부에서 모델 호출용으로 쓰임) | 
| **적합한  사용 사례(Use Case)** | AWS 환경에서 빠르게 프로덕션 배포가 필요한 챗봇, 자율 도구 실행 에이전트 | 복잡한 승인 절차(Human-in-the-loop)나 엄격한 순서 제어가 필요한 시스템 | 마케팅 컨텐츠 작성, 코드 리뷰 등 여러 페르소나의 협업이 필요한 작업 | 하나의 앱에서 OpenAI, Claude, Llama 등을 동적으로 스위칭해야 할 때 | 

---
## 4. 요약 및 시사점 (Graph RAG 관점)

Amazon Neptune을 이용한 Graph RAG 아키텍처를 설계할 때 SDK 선택의 기준은 다음과 같습니다.

* **Strands Agents를 선택해야 할 때:** 복잡한 오케스트레이션 코드를 걷어내고, **"Neptune DB 조회 툴", "Vector DB 검색 툴"을 MCP로 던져주기만 하면 강력한 LLM(Claude 3.5 Sonnet 등)이 알아서 적절한 도구를 조합해 답변을 도출**하는 유연하고 빠른 개발이 필요하다면 Strands Agents가 최고의 선택입니다.

* **참고 (LiteLLM의 역할):** LiteLLM은 Strands Agents와 경쟁하는 기술이 아닙니다. 오히려 Strands 내부에서 다양한 외부 LLM을 호출할 때 사용하는 **'어댑터(Adapter)'** 역할을 하여 두 기술은 훌륭한 시너지를 냅니다.