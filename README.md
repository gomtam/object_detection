# 객체 인식 프로그램

이 프로그램은 YOLO(You Only Look Once) 모델을 사용하여 실시간으로 객체 및 동물을 인식하는 프로그램입니다.

## 프로그램 설명

1. **통합 객체 인식 프로그램 (unified_object_detection.py)**
   - COCO 데이터셋으로 사전 훈련된 YOLOv3 또는 YOLOv3-tiny 모델을 사용
   - 화면 캡처 또는 웹캠에서 객체 인식 가능
   - 80개 클래스의 객체 인식 지원
   - 작은 객체 감지 모드 제공

2. **통합 동물 인식 프로그램 (unified_animal_detection.py)**
   - 객체 인식 프로그램을 기반으로 동물만 검출하도록 특화
   - 10종류의 동물 인식: 새, 고양이, 개, 말, 양, 소, 코끼리, 곰, 얼룩말, 기린
   
3. **카메라 확인 프로그램 (check_cameras.py)**
   - 시스템에 연결된 모든 카메라 장치를 확인
   - 카메라 미리보기와 ID를 표시

## 요구 사항

- Python 3.6 이상
- OpenCV 4.x
- NumPy
- PyAutoGUI
- Pillow (PIL)

## 설치 방법

1. 필요한 패키지 설치:
   ```
   pip install opencv-python numpy pyautogui pillow
   ```

2. YOLO 모델 파일 다운로드:
   * [YOLOv3 가중치](https://pjreddie.com/media/files/yolov3.weights) - 236MB
   * [YOLOv3-tiny 가중치](https://pjreddie.com/media/files/yolov3-tiny.weights) - 33MB
   * [YOLOv3 설정 파일](https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg)
   * [YOLOv3-tiny 설정 파일](https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg)
   * [COCO 클래스 이름](https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names)

3. 다운로드한 파일을 프로그램과 같은 폴더에 저장하세요.

## 사용 방법

### 카메라 장치 확인

프로그램 실행 전에 시스템에 연결된 카메라를 확인할 수 있습니다:

```
python check_cameras.py
```

이 명령어는 연결된 모든 카메라와 그 미리보기를 표시합니다.

### 객체 인식 프로그램 실행

#### 1. 대화형 메뉴 사용 (권장)

객체 인식 프로그램을 실행하면 다음과 같은 대화형 메뉴가 표시됩니다:
```
python unified_object_detection.py
```

메뉴에서 다음 옵션을 선택할 수 있습니다:
- 입력 소스 (화면 캡처 또는 특정 카메라)
- 모델 유형 (YOLOv3-tiny 또는 YOLOv3)
- 감지 모드 (일반 또는 작은 객체 감지 강화)

#### 2. 명령행 옵션 사용

대화형 메뉴 없이 직접 실행할 수도 있습니다:
```
python unified_object_detection.py --direct --webcam --camera-id 1 --no-tiny --small-detection
```

명령행 옵션:
- `--direct`: 대화형 메뉴를 사용하지 않고 직접 실행
- `--webcam`: 화면 캡처 대신 웹캠 사용
- `--camera-id ID`: 사용할 카메라 ID 지정 (기본값: 0)
- `--no-tiny`: YOLOv3-tiny 대신 YOLOv3 모델 사용
- `--small-detection`: 작은 객체 감지 강화 모드 활성화

### 동물 인식 프로그램 실행

#### 1. 대화형 메뉴 사용 (권장)

동물 인식 프로그램을 실행하면 다음과 같은 대화형 메뉴가 표시됩니다:
```
python unified_animal_detection.py
```

객체 인식 프로그램과 동일한 메뉴 옵션이 제공됩니다.

#### 2. 명령행 옵션 사용

대화형 메뉴 없이 직접 실행:
```
python unified_animal_detection.py --direct --webcam --camera-id 1 --no-tiny --small-detection
```

명령행 옵션은 객체 인식 프로그램과 동일합니다.

## 기능

- 객체 인식 (80가지 클래스)
- 동물 인식 (10가지 동물)
- 작은 객체 감지 강화 모드
- 웹캠 또는 화면 캡처 사용 선택
- FPS 및 탐지 시간 표시
- 객체 크기에 따른 표시 방법 차별화
- CUDA 가속 자동 사용 (가능한 경우)

## 문제 해결

1. **카메라가 열리지 않는 경우**
   - 다른 카메라 ID 시도
   - 카메라가 다른 프로그램에서 사용 중인지 확인
   - 카메라 드라이버 업데이트
   - USB 카메라인 경우 다른 USB 포트 시도

2. **프로그램이 느린 경우**
   - YOLOv3-tiny 모델로 전환
   - 작은 객체 감지 모드를 비활성화
   - 화면 캡처 대신 카메라 입력 사용

3. **모델 파일 관련 오류**
   - 필요한 모든 파일이 프로그램과 같은 폴더에 있는지 확인
   - 파일 이름이 정확한지 확인 (대소문자 구분)

## 주의 사항

- 웹캠 해상도는 기본적으로 640x480으로 설정됩니다.
- 화면 캡처 모드는 전체 화면을 캡처합니다.
- CUDA 가속은 NVIDIA GPU와 CUDA가 설치된 경우에만 작동합니다. 