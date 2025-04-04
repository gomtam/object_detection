import cv2
import time
import numpy as np
import os

def check_available_cameras(max_cameras=5):
    """사용 가능한 카메라 장치를 확인하고 정보를 반환합니다."""
    available_cameras = []
    
    print("\n===== 카메라 장치 검색 중 =====")
    
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
                cv2.waitKey(1000)  # 1초 동안 이미지 표시
            else:
                print(f"카메라 {i}: 연결됨 (프레임을 읽을 수 없음)")
        
        # 카메라 자원 해제
        cap.release()
        time.sleep(0.3)  # 다음 카메라 검색 전 잠시 대기
    
    return available_cameras

def main():
    """메인 함수: 사용 가능한 카메라를 확인하고 실시간 미리보기를 제공합니다."""
    os.system('cls' if os.name == 'nt' else 'clear')  # 화면 지우기
    
    print("\n===== 카메라 확인 프로그램 =====")
    print("시스템에 연결된 카메라를 검색합니다...")
    
    available_cameras = check_available_cameras()
    
    if not available_cameras:
        print("\n사용 가능한 카메라가 없습니다.")
        print("다음을 확인하세요:")
        print("1. 카메라가 컴퓨터에 올바르게 연결되어 있는지 확인하세요.")
        print("2. 카메라 드라이버가 설치되어 있는지 확인하세요.")
        print("3. 다른 프로그램이 카메라를 사용 중인지 확인하세요.")
        input("\n종료하려면 엔터 키를 누르세요...")
        return
    
    print(f"\n발견된 카메라: {len(available_cameras)}개")
    
    # 실시간 미리보기 시작
    print("\n각 카메라의 실시간 미리보기를 시작합니다...")
    print("미리보기를 종료하려면 'q' 키를 누르세요.\n")
    
    # 각 카메라에 대한 캡처 객체 초기화
    caps = []
    for camera_id, _, _ in available_cameras:
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            caps.append((camera_id, cap))
    
    try:
        # ESC 또는 'q' 키를 누를 때까지 미리보기 표시
        while True:
            for camera_id, cap in caps:
                ret, frame = cap.read()
                if ret:
                    window_name = f"카메라 {camera_id} (종료: q 키)"
                    cv2.imshow(window_name, frame)
            
            # 키 이벤트 확인 (1ms 대기)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC 또는 'q' 키
                break
    
    finally:
        # 자원 정리
        for _, cap in caps:
            cap.release()
        cv2.destroyAllWindows()
    
    # 사용자 안내
    print("\n카메라 확인이 완료되었습니다.")
    print("객체 인식 프로그램을 실행할 때 다음 명령어를 사용할 수 있습니다:")
    print("\n1. 인터랙티브 메뉴로 실행:")
    print("   python unified_object_detection.py")
    print("   python unified_animal_detection.py")
    print("\n2. 명령줄에서 특정 카메라 지정:")
    print("   python unified_object_detection.py --direct --webcam --camera-id <카메라_ID>")
    print("   python unified_animal_detection.py --direct --webcam --camera-id <카메라_ID>")
    print("\n<카메라_ID>는 위에 표시된 카메라 번호로 대체하세요.")

if __name__ == "__main__":
    main() 