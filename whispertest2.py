import pyaudio
import numpy as np
import time
import threading
import queue
from faster_whisper import WhisperModel
import io

class SimpleRealtimeWhisper:
    def __init__(self):
        """간단한 실시간 음성인식 클래스"""
        print("🎤 실시간 음성인식 초기화 중...")
        
        # 오디오 설정
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        
        # 오디오 버퍼
        self.audio_buffer = queue.Queue()
        self.is_recording = False
        
        # Whisper 모델 로드 (가장 빠른 tiny 모델 사용)
        print("Whisper 모델 로딩 중... (처음 실행시 다운로드 시간이 걸릴 수 있습니다)")
        self.model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("✅ 모델 로딩 완료!")
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        """오디오 입력 콜백"""
        if self.is_recording:
            self.audio_buffer.put(in_data)
        return (in_data, pyaudio.paContinue)
    
    def process_audio(self, audio_data):
        """오디오를 텍스트로 변환"""
        try:
            # numpy 배열로 변환
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_np.astype(np.float32) / 32768.0
            
            # 너무 짧으면 무시
            if len(audio_float) < self.RATE * 1.0:  # 1초 미만
                return None
                
            # Whisper로 전사
            segments, info = self.model.transcribe(
                audio_float,
                language="ko",
                beam_size=1,
                best_of=1
            )
            
            # 결과 수집
            text = ""
            for segment in segments:
                if segment.text.strip():
                    text += segment.text.strip() + " "
            
            return text.strip()
            
        except Exception as e:
            print(f"❌ 오디오 처리 오류: {e}")
            return None
    
    def start_recording(self):
        """녹음 시작"""
        self.p = pyaudio.PyAudio()
        
        # 오디오 스트림 시작
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self.audio_callback
        )
        
        self.is_recording = True
        self.stream.start_stream()
        
        print("\n" + "="*50)
        print("🎤 실시간 음성인식 시작!")
        print("💬 마이크에 대고 말해보세요")
        print("⏹️  종료하려면 Ctrl+C를 누르세요")
        print("="*50)
        
        # 오디오 처리 스레드
        def process_audio_thread():
            audio_chunk = b""
            chunk_duration = 3.0  # 3초마다 처리
            chunk_size = int(self.RATE * chunk_duration)
            
            while self.is_recording:
                try:
                    # 오디오 데이터 수집
                    if not self.audio_buffer.empty():
                        audio_chunk += self.audio_buffer.get_nowait()
                    
                    # 충분한 데이터가 모이면 처리
                    if len(audio_chunk) >= chunk_size:
                        process_data = audio_chunk[:chunk_size]
                        audio_chunk = audio_chunk[chunk_size:]
                        
                        # 텍스트 변환
                        text = self.process_audio(process_data)
                        if text:
                            timestamp = time.strftime("%H:%M:%S")
                            print(f"🕐 [{timestamp}] {text}")
                            
                except queue.Empty:
                    time.sleep(0.01)
                except Exception as e:
                    print(f"❌ 처리 오류: {e}")
                    time.sleep(0.1)
        
        # 처리 스레드 시작
        self.process_thread = threading.Thread(target=process_audio_thread, daemon=True)
        self.process_thread.start()
    
    def stop_recording(self):
        """녹음 중지"""
        print("\n🛑 녹음을 중지합니다...")
        self.is_recording = False
        
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        
        if hasattr(self, 'p'):
            self.p.terminate()
        
        print("✅ 녹음이 중지되었습니다.")

def main():
    """메인 함수"""
    try:
        # 실시간 음성인식 시작
        whisper = SimpleRealtimeWhisper()
        whisper.start_recording()
        
        # 사용자가 Ctrl+C를 누를 때까지 대기
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n👋 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        if 'whisper' in locals():
            whisper.stop_recording()

if __name__ == "__main__":
    main()