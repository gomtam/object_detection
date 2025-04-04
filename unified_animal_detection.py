import cv2
import numpy as np
import time
import pyautogui
from PIL import ImageGrab
import argparse
import os

# 동물 클래스 목록 (COCO 데이터셋 기준)
ANIMAL_CLASSES = {
    'bird': 14,     # 새
    'cat': 15,      # 고양이
    'dog': 16,      # 개
    'horse': 17,    # 말
    'sheep': 18,    # 양
    'cow': 19,      # 소
    'elephant': 20, # 코끼리
    'bear': 21,     # 곰
    'zebra': 22,    # 얼룩말
    'giraffe': 23   # 기린
}

def check_available_cameras(max_cameras=5):
    """사용 가능한 카메라 장치를 확인하고 정보를 반환합니다."""
    available_cameras = []
    
    print("사용 가능한 카메라 장치 검색 중...")
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap is None or not cap.isOpened():
            print(f"카메라 {i}: 사용할 수 없음")
        else:
            # 카메라 정보 가져오기
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # 한 프레임 읽기 시도
            ret, frame = cap.read()
            if ret:
                print(f"카메라 {i}: 사용 가능 - 해상도: {width}x{height}, FPS: {fps:.1f}")
                # 카메라 이름 설정
                if i == 0:
                    camera_name = "내장 카메라"
                else:
                    camera_name = f"외부 카메라 {i}"
                    
                available_cameras.append((i, camera_name, frame))
                
                # 카메라 미리보기 표시
                window_name = f"카메라 {i} 미리보기"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 320, 240)
                cv2.imshow(window_name, frame)
                cv2.waitKey(800)  # 0.8초 동안 이미지 표시
                cv2.destroyWindow(window_name)
            else:
                print(f"카메라 {i}: 연결됨 (프레임을 읽을 수 없음)")
        
        # 카메라 자원 해제
        cap.release()
        time.sleep(0.3)  # 다음 카메라 검색 전 잠시 대기
    
    return available_cameras

def show_menu():
    """사용자 메뉴를 표시하고 선택값을 반환합니다."""
    os.system('cls' if os.name == 'nt' else 'clear')  # 화면 지우기
    
    print("\n===== 동물 인식 프로그램 =====")
    print("\n인식 가능한 동물:", ", ".join(ANIMAL_CLASSES.keys()))
    print("\n1. 입력 소스 선택")
    
    # 카메라 검색
    available_cameras = check_available_cameras()
    
    # 입력 소스 옵션 표시
    print("\n[입력 소스 옵션]")
    print("0: 화면 캡처")
    
    camera_options = {}
    for idx, (cam_id, cam_name, _) in enumerate(available_cameras, 1):
        print(f"{idx}: {cam_name} (카메라 ID: {cam_id})")
        camera_options[idx] = cam_id
    
    # 사용자 입력 받기
    while True:
        try:
            source_choice = int(input("\n입력 소스를 선택하세요 (숫자): "))
            if source_choice == 0:
                use_webcam = False
                camera_id = None
                break
            elif source_choice in camera_options:
                use_webcam = True
                camera_id = camera_options[source_choice]
                break
            else:
                print("잘못된 선택입니다. 다시 시도하세요.")
        except ValueError:
            print("숫자를 입력하세요.")
    
    # 모델 선택
    print("\n2. 모델 선택")
    print("\n[모델 옵션]")
    print("1: YOLOv3-tiny (빠름, 정확도 낮음)")
    print("2: YOLOv3 (느림, 정확도 높음)")
    
    while True:
        try:
            model_choice = int(input("\n모델을 선택하세요 (숫자): "))
            if model_choice in [1, 2]:
                tiny_model = (model_choice == 1)
                break
            else:
                print("잘못된 선택입니다. 다시 시도하세요.")
        except ValueError:
            print("숫자를 입력하세요.")
    
    # 작은 객체 감지 모드 선택
    print("\n3. 작은 객체 감지 모드 선택")
    print("\n[감지 모드 옵션]")
    print("1: 일반 모드 (빠름)")
    print("2: 작은 객체 감지 강화 모드 (느림, 작은 객체 인식 향상)")
    
    while True:
        try:
            detection_choice = int(input("\n감지 모드를 선택하세요 (숫자): "))
            if detection_choice in [1, 2]:
                small_detection = (detection_choice == 2)
                break
            else:
                print("잘못된 선택입니다. 다시 시도하세요.")
        except ValueError:
            print("숫자를 입력하세요.")
    
    # 선택 정보 표시
    print("\n[선택한 설정]")
    print(f"입력 소스: {'화면 캡처' if not use_webcam else f'카메라 (ID: {camera_id})'}")
    print(f"모델: {'YOLOv3-tiny' if tiny_model else 'YOLOv3'}")
    print(f"감지 모드: {'일반 모드' if not small_detection else '작은 객체 감지 강화 모드'}")
    
    input("\n엔터 키를 눌러 계속하세요...")
    
    return use_webcam, camera_id, tiny_model, small_detection

def unified_animal_detection(use_webcam=False, camera_id=0, tiny_model=True, small_detection=True):
    # 입력 소스 설정
    if use_webcam:
        # 웹캠 초기화
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"오류: 카메라 {camera_id}를 열 수 없습니다.")
            print("다른 카메라를 선택하세요.")
            input("종료하려면 엔터 키를 누르세요...")
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        def get_frame():
            ret, frame = cap.read()
            if not ret:
                print("카메라에서 영상을 가져올 수 없습니다.")
                return None
            return frame
            
        input_source_name = f"웹캠 (카메라 ID: {camera_id})"
    else:
        # 화면 크기 설정
        screen_width, screen_height = pyautogui.size()
        
        def get_frame():
            # 전체 화면 캡처
            img = ImageGrab.grab(bbox=(0, 0, screen_width, screen_height))
            img = np.array(img)
            # RGB to BGR 변환 (OpenCV는 BGR 형식 사용)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
            
        input_source_name = "화면"
    
    # 창 크기 설정 (원래 크기의 3분의 1)
    window_name = f"동물 인식 ({input_source_name})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    if use_webcam:
        display_width = int(640 / 3)
        display_height = int(480 / 3)
    else:
        display_width = int(screen_width / 3)
        display_height = int(screen_height / 3)
    
    cv2.resizeWindow(window_name, display_width, display_height)
    
    # 작은 물체 감지 모드 여부 표시
    mode_text = "작은 객체 감지 강화 모드" if small_detection else "일반 모드"
    model_type = "YOLOv3-tiny" if tiny_model else "YOLOv3"
    
    print(f"동물 인식 프로그램이 시작되었습니다. 입력 소스: {input_source_name}")
    print(f"모델: {model_type}, 모드: {mode_text}")
    print(f"인식 가능한 동물: {', '.join(ANIMAL_CLASSES.keys())}")
    print("종료하려면 'q'를 누르세요.")
    
    # YOLO 모델 설정 (tiny 또는 일반)
    if tiny_model:
        config_file = "yolov3-tiny.cfg"
        weights_file = "yolov3-tiny.weights"
    else:
        config_file = "yolov3.cfg"
        weights_file = "yolov3.weights"
    
    # 성능 개선을 위한 변수들
    frame_count = 0
    skip_frames = 2  # 이 값을 높이면 더 많은 프레임을 건너뜁니다
    
    # 작은 객체 감지를 위한 파라미터 조정
    if small_detection:
        confidence_threshold = 0.25  # 작은 객체 감지를 위해 임계값 낮춤
        nms_threshold = 0.3  # NMS 임계값 조정
        input_size = (608, 608)  # 더 큰 입력 크기로 작은 객체 감지 향상
    else:
        confidence_threshold = 0.4
        nms_threshold = 0.4
        input_size = (416, 416)
    
    try:
        # GPU 가속 시도
        net = cv2.dnn.readNetFromDarknet(config_file, weights_file)
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("CUDA 가속이 활성화되었습니다.")
        except:
            print("CUDA 가속을 사용할 수 없습니다. CPU 모드로 실행됩니다.")
        
        layer_names = net.getLayerNames()
        try:
            # OpenCV 4.5.4 이상
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        except:
            # OpenCV 4.5.3 이하
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
        # COCO 클래스 로드
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        
        # 동물 클래스 ID 리스트 생성
        animal_class_ids = list(ANIMAL_CLASSES.values())
        
        last_time = time.time()
        fps = 0
        animal_counts = {name: 0 for name in ANIMAL_CLASSES.keys()}
        
        # 멀티스케일 감지를 위한 추가 설정
        multi_scales = [0.5, 0.75, 1.0] if small_detection else [1.0]
        
        while True:
            # 프레임 가져오기
            frame = get_frame()
            if frame is None:
                break
            
            # 프레임 스킵 (프레임 처리 빈도 줄이기)
            frame_count += 1
            if frame_count % skip_frames != 0:
                # 프레임을 건너뛰더라도 화면은 표시
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # FPS 계산
            current_time = time.time()
            if current_time - last_time > 0:
                fps = 1 / (current_time - last_time)
            last_time = current_time
            
            # 이미지 크기 축소 (처리 속도 향상)
            small_frame = cv2.resize(frame, input_size)
            
            # 이미지 전처리
            blob = cv2.dnn.blobFromImage(small_frame, 1/255.0, input_size, swapRB=True, crop=False)
            
            # 모델에 이미지 입력
            net.setInput(blob)
            
            # 예측 실행
            start_time = time.time()
            outputs = net.forward(output_layers)
            end_time = time.time()
            
            # 예측 시간 표시
            model_fps = 1/(end_time-start_time)
            fps_text = f"FPS: {fps:.2f}, 모델 FPS: {model_fps:.2f}"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 모드 표시
            mode_text = f"모드: {'작은 객체 감지 강화' if small_detection else '일반'}"
            cv2.putText(frame, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 객체 인식 결과 처리
            class_ids = []
            confidences = []
            boxes = []
            
            height, width, _ = frame.shape
            
            # 임시 동물 카운트 초기화
            temp_animal_counts = {name: 0 for name in ANIMAL_CLASSES.keys()}
            
            # 탐지된 객체 정보 수집
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # 동물 클래스이고 임계값 이상인 경우만 처리
                    if class_id in animal_class_ids and confidence > confidence_threshold:
                        # 객체 좌표 계산
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # 작은 객체 감지를 위한 크기 조정
                        if small_detection and w * h < (width * height * 0.01):  # 전체 화면의 1% 미만
                            confidence *= 1.2  # 작은 객체에 대한 신뢰도 가중치 부여
                            confidence = min(confidence, 1.0)  # 1.0을 초과하지 않도록
                        
                        # 좌표 계산
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
                        
                        # 동물 종류별 카운트 증가
                        for animal_name, animal_id in ANIMAL_CLASSES.items():
                            if class_id == animal_id:
                                temp_animal_counts[animal_name] += 1
            
            # Non-maximum suppression 적용
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
            
            # 동물 카운트 업데이트
            animal_counts = temp_animal_counts
            
            # 인식된 동물 표시
            if len(boxes) > 0:
                if isinstance(indexes, tuple):
                    indexes = indexes[0]
                
                for i in indexes:
                    if isinstance(i, list):
                        i = i[0]
                    
                    x, y, w, h = boxes[i]
                    class_id = class_ids[i]
                    
                    # 동물 이름 찾기
                    animal_name = None
                    for name, c_id in ANIMAL_CLASSES.items():
                        if class_id == c_id:
                            animal_name = name
                            break
                    
                    if animal_name:
                        label = f"{animal_name}: {confidences[i]:.2f}"
                        
                        # 객체 크기에 따른 색상 결정 (작은 객체는 다른 색상으로 표시)
                        is_small_object = (w * h) < (width * height * 0.01)  # 전체 화면의 1% 미만이면 작은 객체
                        
                        # 각 동물 종류마다 다른 색상 사용
                        color_map = {
                            'bird': (0, 255, 0),      # 녹색
                            'cat': (255, 0, 0),       # 파란색
                            'dog': (0, 0, 255),       # 빨간색
                            'horse': (255, 255, 0),   # 청록색
                            'sheep': (0, 255, 255),   # 노란색
                            'cow': (255, 0, 255),     # 분홍색
                            'elephant': (128, 0, 128), # 보라색
                            'bear': (128, 128, 0),    # 올리브색
                            'zebra': (0, 128, 128),   # 티얼색
                            'giraffe': (128, 0, 0)    # 갈색
                        }
                        
                        base_color = color_map.get(animal_name, (0, 255, 0))
                        
                        # 작은 객체일 경우 하이라이트
                        if is_small_object and small_detection:
                            # 바운딩 박스 더 두껍게 그리기
                            thickness = 3
                            # 색상 더 밝게 (255, 255, 255)에 가깝게
                            color = tuple([min(c + 100, 255) for c in base_color])
                            # 작은 객체 표시
                            cv2.putText(frame, "[작은 객체]", (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        else:
                            thickness = 2
                            color = base_color
                        
                        # 바운딩 박스 그리기
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 동물 카운트 정보 표시
            y_pos = 90
            cv2.putText(frame, "발견된 동물:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 30
            
            for animal_name, count in animal_counts.items():
                if count > 0:
                    cv2.putText(frame, f"{animal_name}: {count}마리", (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_pos += 25
            
            # 결과 표시
            cv2.imshow(window_name, frame)
            
            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"오류: {e}")
        print(f"모델 파일이 필요합니다: {config_file}, {weights_file}")
        print("README.md 파일을 참조하세요.")
    
    # 자원 해제
    if use_webcam:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 명령행 인자로 직접 실행 모드 활성화 가능
    parser = argparse.ArgumentParser(description='통합 동물 인식 프로그램')
    parser.add_argument('--direct', action='store_true', help='메뉴 없이 직접 실행 (아래 옵션 활성화)')
    parser.add_argument('--webcam', action='store_true', help='웹캠 사용 (기본값: 화면 캡처)')
    parser.add_argument('--camera-id', type=int, default=0, help='사용할 카메라 ID (기본값: 0)')
    parser.add_argument('--no-tiny', action='store_true', help='YOLOv3 모델 사용 (기본값: YOLOv3-tiny)')
    parser.add_argument('--small-detection', action='store_true', help='작은 객체 감지 강화 모드 활성화 (기본값: 활성화)')
    parser.add_argument('--no-small-detection', action='store_true', help='작은 객체 감지 강화 모드 비활성화')
    args = parser.parse_args()
    
    if args.direct:
        # 직접 명령어로 실행
        # 작은 객체 감지 모드 결정 (기본값: 활성화, --no-small-detection 옵션으로 비활성화)
        small_detection = not args.no_small_detection
        
        unified_animal_detection(
            use_webcam=args.webcam,
            camera_id=args.camera_id,
            tiny_model=not args.no_tiny,
            small_detection=small_detection
        )
    else:
        # 대화형 메뉴로 실행
        use_webcam, camera_id, tiny_model, small_detection = show_menu()
        unified_animal_detection(
            use_webcam=use_webcam,
            camera_id=camera_id if camera_id is not None else 0,
            tiny_model=tiny_model,
            small_detection=small_detection
        ) 