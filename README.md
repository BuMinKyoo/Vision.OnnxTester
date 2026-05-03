# Vision.OnnxTester

WPF (.NET 10) 기반의 ONNX Runtime 학습용 테스트 프로젝트.
YOLOv8 모델을 로드해 정적 이미지(CCTV 스냅샷 등)에서 객체를 검출하고, Bounding Box를 시각화한다.

> 학습 목표: AI 모델의 입출력인 **텐서(Tensor)** 와 **다차원 배열 구조** 를 직접 다뤄본다.

---

## 1. 개발 환경

| 항목 | 버전 / 비고 |
|---|---|
| OS | Windows 11 |
| IDE | Visual Studio 2022 (또는 Rider) |
| .NET SDK | .NET 10 (현재 csproj: `net10.0-windows`) |
| 언어 | C# (Nullable enable, ImplicitUsings enable) |
| UI | WPF |

---

## 2. 코딩 전 준비물

### 2-1. NuGet 패키지

Visual Studio의 **NuGet 패키지 관리자** 에서 아래 두 개를 설치한다.

| 패키지 | 용도 |
|---|---|
| `Microsoft.ML.OnnxRuntime` | ONNX 모델 로딩 및 추론 (CPU 버전) |
| `SixLabors.ImageSharp` | 이미지 픽셀 단위 조작 (리사이즈, 채널 추출, 정규화) |

> GPU(NVIDIA + CUDA)가 있다면 `Microsoft.ML.OnnxRuntime` 대신
> `Microsoft.ML.OnnxRuntime.Gpu` 를 설치하면 된다. 일단은 CPU로 시작.

설치 후 `Vision.OnnxTester.csproj` 에 다음과 같이 추가될 예정이다.

```xml
<ItemGroup>
  <PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.x.x" />
  <PackageReference Include="SixLabors.ImageSharp" Version="3.x.x" />
</ItemGroup>
```

### 2-2. YOLOv8 ONNX 모델 파일

가장 가볍고 빠른 **`yolov8n.onnx`** (Nano 버전, 약 12 MB) 를 사용한다.

- 입력 형태: `float32[1, 3, 640, 640]` — Batch=1, RGB 3채널, 640x640
- 출력 형태: `float32[1, 84, 8400]` — 8400개 후보 박스 각각에 (x, y, w, h, class0~class79 confidence) = 4 + 80 = 84

> **이 저장소에는 `yolov8n.pt` 와 변환된 `yolov8n.onnx` 가 이미 포함되어 있다.**
> 클론만 하면 바로 사용 가능하므로 아래 다운로드/변환 절차는 **참고용**.

#### 공식 다운로드 / 변환 경로

| 자료 | 링크 |
|---|---|
| Ultralytics 공식 GitHub (메인) | <https://github.com/ultralytics/ultralytics> |
| `.pt` 모델 파일 직접 다운로드 (Release Assets) | <https://github.com/ultralytics/assets/releases> |
| Hugging Face Ultralytics 미러 | <https://huggingface.co/Ultralytics/YOLOv8> |
| ONNX export 공식 가이드 | <https://docs.ultralytics.com/modes/export/> |

#### `.pt` → `.onnx` 변환 명령

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx imgsz=640
```

생성된 `yolov8n.onnx` 는 다음 위치에 둔다.

```
Vision.OnnxTester/
└── Assets/
    └── Models/
        ├── yolov8n.pt          <-- 원본 (선택)
        └── yolov8n.onnx        <-- 추론에 사용
```

> `.csproj` 에서 `<None Update="Assets\Models\yolov8n.onnx" CopyToOutputDirectory="PreserveNewest" />`
> 로 빌드 출력 폴더에 자동 복사되도록 설정한다.

### 2-3. 테스트용 이미지

사람, 차량 등이 찍힌 JPG/PNG 이미지를 한 장 준비한다.

```
Vision.OnnxTester/
└── Assets/
    └── TestImages/
        └── cctv_sample.jpg      <-- 여기
```

---

## 3. 핵심 개념: 텐서(Tensor)

YOLOv8 입력 텐서는 4차원 배열 `[1, 3, 640, 640]` 이다.

| 축 | 크기 | 의미 |
|---|---|---|
| 0 (Batch) | 1 | 한 번에 처리할 이미지 개수 |
| 1 (Channel) | 3 | 색상 채널 (R, G, B) |
| 2 (Height) | 640 | 이미지 세로 |
| 3 (Width) | 640 | 이미지 가로 |

JPG 이미지를 이 텐서로 변환하기까지의 과정:

1. **Letterbox 리사이즈**: 가로세로 비율을 유지하면서 640x640 캔버스에 맞추고 빈 공간은 회색(114,114,114)으로 패딩
2. **색공간 정렬**: BGR(혹은 RGBA) → RGB 순서로
3. **정규화**: 픽셀값 0~255 → 0.0~1.0 (`/ 255.0f`)
4. **차원 재배열 (HWC → CHW)**: ImageSharp는 픽셀을 `(Height, Width, Channel)` 순으로 가지고 있는데, ONNX는 `(Channel, Height, Width)` 를 요구함

이 4단계가 Step 3의 가장 중요한 학습 포인트.

---

## 4. 진행 단계 (Step 3 세부 로드맵)

| # | 작업 | 결과물 |
|---|---|---|
| 1 | NuGet 패키지 설치 | `csproj` 수정 |
| 2 | `Assets/Models`, `Assets/TestImages` 폴더 생성 + 파일 배치 | 자산 배치 |
| 3 | `Models/Detection.cs` — BBox/Class/Score DTO | 결과 표현 클래스 |
| 4 | `Services/CocoLabels.cs` — COCO 80 클래스 이름 매핑 | 라벨 매핑 |
| 5 | `Services/ImagePreprocessor.cs` — JPG → `DenseTensor<float>` | 전처리 로직 |
| 6 | `Services/YoloV8Detector.cs` — `InferenceSession` 로드 + 추론 + NMS 후처리 | 검출 엔진 |
| 7 | `MainWindow.xaml(.cs)` — 이미지 선택 버튼, 결과 박스 오버레이 | UI |

---

## 5. 폴더 구조 (목표)

```
Vision.OnnxTester/
├── App.xaml
├── App.xaml.cs
├── MainWindow.xaml
├── MainWindow.xaml.cs
├── Vision.OnnxTester.csproj
├── Assets/
│   ├── Models/
│   │   └── yolov8n.onnx
│   └── TestImages/
│       └── cctv_sample.jpg
├── Models/
│   └── Detection.cs
└── Services/
    ├── CocoLabels.cs
    ├── ImagePreprocessor.cs
    └── YoloV8Detector.cs
```
