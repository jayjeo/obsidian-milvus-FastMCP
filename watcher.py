import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ObsidianWatcher(FileSystemEventHandler):
    def __init__(self, processor):
        self.processor = processor
        self.last_modified = {}
        
    def on_modified(self, event):
        """파일 수정 감지"""
        if event.is_directory:
            return
        
        if event.src_path.endswith(('.md', '.pdf')):
            # 파일이 정리될 시간을 주기 위해 약간의 지연
            current_time = time.time()
            if event.src_path in self.last_modified and current_time - self.last_modified[event.src_path] < 2:
                return
            
            self.last_modified[event.src_path] = current_time
            time.sleep(1)  # 파일 IO가 완료될 시간 대기
            
            try:
                print(f"File modified: {event.src_path}")
                self.processor.process_file(event.src_path)
            except Exception as e:
                print(f"Error processing modified file: {e}")
    
    def on_created(self, event):
        """파일 생성 감지"""
        if event.is_directory:
            return
        
        if event.src_path.endswith(('.md', '.pdf')):
            time.sleep(1)  # 파일 IO가 완료될 시간 대기
            
            try:
                print(f"File created: {event.src_path}")
                self.processor.process_file(event.src_path)
            except Exception as e:
                print(f"Error processing new file: {e}")
    
    def on_deleted(self, event):
        """파일 삭제 감지"""
        if event.is_directory:
            return
        
        if event.src_path.endswith(('.md', '.pdf')):
            try:
                rel_path = os.path.relpath(
                    event.src_path, 
                    os.path.expanduser(self.processor.vault_path)
                )
                print(f"File deleted: {rel_path}")
                self.processor.milvus_manager.delete_by_path(rel_path)
            except Exception as e:
                print(f"Error handling deleted file: {e}")


def start_watcher(processor):
    """Obsidian 볼트 변경 감지 시작"""
    try:
        event_handler = ObsidianWatcher(processor)
        observer = Observer()
        
        # 경로가 존재하는지 확인
        vault_path = processor.vault_path
        if not os.path.exists(vault_path):
            print(f"경고: 볼트 경로를 찾을 수 없습니다: {vault_path}")
            return
            
        observer.schedule(
            event_handler, 
            vault_path,
            recursive=True
        )
        observer.start()
        print(f"Started watching Obsidian vault at {processor.vault_path}")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        
        observer.join()
    except KeyboardInterrupt:
        observer.stop()
    
    observer.join()