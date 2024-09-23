import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import copy
from collections import defaultdict
import numpy as np
import time
import random
class AnnotationToolApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Annotation Tool")
        self.rest_duration = 180  # 3 minutes in seconds
        self.last_annotation_time = time.time()  # To track the last annotation time
        self.canvas = tk.Canvas(root, width=800, height=700)
        self.canvas.pack()
        self.h = None
        self.w = None

        self.frame_label = tk.Label(root, text="", font=("Helvetica", 16))
        self.frame_label.pack(fill=tk.BOTH, expand=True)

        self.occlusion_var = tk.BooleanVar()
        self.occlusion_var.set(False)
        self.occ_count = []
        self.occ_flag = False
        self.occlusion_checkbox = tk.Checkbutton(root, text="Occlusion", variable=self.occlusion_var)
        self.occlusion_checkbox.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.out_of_view_var = tk.BooleanVar()
        self.out_of_view_var.set(False)
        self.oov_count = []
        self.oov_flag = False
        self.out_of_view_checkbox = tk.Checkbutton(root, text="Out of View", variable=self.out_of_view_var)
        self.out_of_view_checkbox.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.save_var = tk.BooleanVar()
        self.save_var.set(False)
        self.save_checkbox = tk.Checkbutton(root, text="Save", variable=self.save_var)
        self.save_checkbox.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.skip_button = tk.Button(root, text="Skip", command=self.skip_sequence)
        self.skip_button.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.jump_button = tk.Button(root, text="Jump", command=self.jump_to_frame)
        self.jump_button.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        
        self.frame_jump_entry = tk.Entry(root)
        self.frame_jump_entry.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        
        self.clear_button = tk.Button(root, text="Clear", command=self.clear_annotation)
        self.clear_button.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        
        self.sequences = []
        self.anno = defaultdict(lambda: -1)
        self.occ = defaultdict(lambda: -1)
        self.oov = defaultdict(lambda: -1)
        self.imgs = defaultdict(lambda: -1)
        self.seq_idx = 0
        self.current_seq_idx = 0
        self.cur_frame_idx = 0
        self.tmp_flag = True
        self.cap = None
        self.original_img = None
        self.drawing = False
        self.start_x, self.start_y = -1, -1
        self.end_x, self.end_y = -1, -1
        self.sequence_name = None
        self.Anno_path = None
        self.path = filedialog.askdirectory()

        self.get_video_files()
        self.sequence_name = self.sequences[self.seq_idx][0]
        self.show_frame(self.sequences[self.seq_idx])
        os.system("cls")
        self.root.bind('a', self.prev_frame)
        self.root.bind('d', self.next_frame)
        self.canvas.bind('<Button-1>', self.on_mouse)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse)
        #self.root.bind('w', self.push_space)
         
    def remove_entry_focus(self):
        self.root.focus_set()
    def get_video_files(self):
        files = os.listdir(self.path)
        for file in files:
            if file.endswith('.mp4') or file.endswith('.avi'):
                video_path = os.path.join(self.path, file)
                sequence = os.path.splitext(file)[0]
                self.sequences.append((sequence, video_path))

    def annotation_exists(self):
        self.Anno_path = os.path.join(self.path, "anno")
        annotation_file_path = os.path.join(self.Anno_path, self.sequence_name + ".txt")
        return os.path.exists(annotation_file_path)

    def update_sequence_list(self):
        self.Anno_path = os.path.join(self.path, "anno")
        if self.annotation_exists():
            return True
        else:
            return False
    def re_obtain_the_results(self):
        print(self.path,self.sequence_name)
        self.Anno_path = os.path.join(self.path, "anno")
        annotation_file_path = os.path.join(self.Anno_path, self.sequence_name + ".txt")

        if self.annotation_exists():
            # Read annotation file
            with open(annotation_file_path, "r") as f:
                for kkkk,line in enumerate(f):
                    values = line.strip().split()
                    print("bounding box:",values)
                    if len(values) == 4:
                        x, y, w, h = map(int, values)
                        self.anno[kkkk] = [x, y, w, h]
            #os.system("pause")
            # Read full occlusion file if it exists
            full_occlusion_file_path = os.path.join(self.path, "full_occlusion", self.sequence_name + "_full_occlusion.txt")
            if os.path.exists(full_occlusion_file_path):
                with open(full_occlusion_file_path, "r") as f:
                    full_occlusion_data = f.read()
                    #print("occ",full_occlusion_data.split(","))
                    for i,j in enumerate(full_occlusion_data.split(",")):
                        self.occ[i] = int(j[0])
                    #self.occ[self.cur_frame_idx] = [int(val) for val in full_occlusion_data.split(",")]

            # Read out of view file if it exists
            out_of_view_file_path = os.path.join(self.path, "out_of_view", self.sequence_name + "_out_of_view.txt")
            if os.path.exists(out_of_view_file_path):
                with open(out_of_view_file_path, "r") as f:
                    out_of_view_data = f.read()
                    #print("oov:",out_of_view_data.split(","))
                    #self.oov[self.cur_frame_idx] = [int(val) for val in out_of_view_data.split(",")]
                    for i,j in enumerate(out_of_view_data.split(",")):
                        self.oov[i] = int(j[0])

    def show_frame(self, sequence_data):
        if self.cur_frame_idx == 0 and self.annotation_exists():
            if self.current_seq_idx ==0 and self.tmp_flag:
                self.tmp_flag = False
                self.re_obtain_the_results()
            elif self.current_seq_idx!= self.seq_idx:
                self.current_seq_idx = self.seq_idx
                self.re_obtain_the_results()
        sequence, video_path = sequence_data
        self.sequence_name = sequence
        if self.cap is None:
            self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            print('Error opening video file')
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cur_frame_idx)

        ret, frame = self.cap.read()
        if ret:
            self.original_img = copy.deepcopy(frame)
            max_canvas_width = self.canvas.winfo_width()
            max_canvas_height = self.canvas.winfo_height()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame, (max_canvas_width, max_canvas_height))

            if self.anno[self.cur_frame_idx] != -1:
                x1, y1, w, h = self.anno[self.cur_frame_idx]
                # Adjust annotation coordinates for display on resized image
                x1_display = int(x1 / frame.shape[1] * frame_resized.shape[1])
                y1_display = int(y1 / frame.shape[0] * frame_resized.shape[0])
                w_display = int(w / frame.shape[1] * frame_resized.shape[1])
                h_display = int(h / frame.shape[0] * frame_resized.shape[0])
                cv2.rectangle(frame_resized, (x1_display, y1_display), (x1_display + w_display, y1_display + h_display), (0, 255, 0), 2)
            else:
                del self.anno[self.cur_frame_idx]
            image = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
            self.canvas.image = image
            self.canvas.create_image(0, 0, image=image, anchor=tk.NW)
            self.canvas.update()

            flag = self.update_sequence_list()
            if flag:
                sequence = sequence + " Already Processed!"
            self.frame_label.config(text=f"{self.cur_frame_idx + 1}/{int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))} {sequence}")
        current_time = time.time()
        if current_time - self.last_annotation_time >= self.rest_duration:
            messagebox.showinfo("Take a Break", "You have labeled for 3 minutes. Take a rest!")
            self.last_annotation_time = current_time
    
    
    def on_mouse(self, event):
        if event.num == 1 and not self.drawing:
            self.start_x, self.start_y = event.x, event.y
            self.drawing = True
        elif event.num == 1 and self.drawing:
            self.end_x, self.end_y = event.x, event.y
            self.drawing = False
            self.h = abs(self.end_y - self.start_y)
            self.w = abs(self.end_x - self.start_x)
            top_left = [min(self.start_x, self.end_x), min(self.start_y, self.end_y)]
            bottom_right = [max(self.start_x, self.end_x), max(self.start_y, self.end_y)]

            # Calculate scaling factors
            img_width, img_height = self.original_img.shape[1], self.original_img.shape[0]
            canvas_width, canvas_height = self.canvas.winfo_width(), self.canvas.winfo_height()
            width_scale = img_width / canvas_width
            height_scale = img_height / canvas_height

            # Adjust annotation coordinates for resizing
            current_anno = [
                int(top_left[0] * width_scale),
                int(top_left[1] * height_scale),
                int((bottom_right[0] - top_left[0]) * width_scale),
                int((bottom_right[1] - top_left[1]) * height_scale)
            ]

            self.save_var.set(True)

            if self.save_var.get():
                self.anno[self.cur_frame_idx] = current_anno
                self.occ[self.cur_frame_idx] = 1 if self.occlusion_var.get() else 0
                self.oov[self.cur_frame_idx] = 1 if self.out_of_view_var.get() else 0
                self.imgs[self.cur_frame_idx] = self.original_img
                os.system("cls")
                print("#" * 10, "frame:", self.cur_frame_idx + 1, "#" * 10)
                print("anno", self.anno[self.cur_frame_idx])
                print("occ", self.occ[self.cur_frame_idx])
                print("oov", self.oov[self.cur_frame_idx])
                tmp_list = [x+1 for x,y in self.anno.items() if y!=-1]
                print("annotated frame:",tmp_list)
                print("#" * 10, "total labeled:", len(tmp_list), "#" * 10)
                print("Not annotated frame:",[kk+1 for kk in range(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))) if kk+1 not in tmp_list])
                #for i,j in self.occ.items():
                #    if j!= -1:
                #        print(i+1," : ",j)
                #for i,j in self.oov.items():
                #    if j!= -1:
                #        print(i+1," : ",j)
                #print(self.occ)
                #print(self.oov)
                #os.system("pause")
            if self.occlusion_var.get():
                self.occ_count.append(1)
            else:
                self.occ_count.append(0)
            if self.out_of_view_var.get():
                self.oov_count.append(1)
            else:
                self.oov_count.append(0)
            if sum(self.occ_count[-3:]) == 3:
                self.occ_flag = True
            else:
                self.occ_flag = False
            if sum(self.oov_count[-3:]) == 3:
                self.oov_flag = True
            else:
                self.oov_flag=False
            self.show_frame(self.sequences[self.seq_idx])
            self.save_var.set(False)
            
    def push_space(self, event):
        pass
    
    def clear_annotation(self):
        self.anno[self.cur_frame_idx] = -1
        self.occ[self.cur_frame_idx] = -1
        self.oov[self.cur_frame_idx] = -1
        self.imgs[self.cur_frame_idx] = -1
        if not self.oov_flag:
            self.out_of_view_var.set(False)
        else:
            self.out_of_view_var.set(True)
        if not self.occ_flag:
            self.occlusion_var.set(False)
        else:
            self.occlusion_var.set(True)
        self.show_frame(self.sequences[self.seq_idx])
        os.system("cls")
        print("#" * 10, "frame:", self.cur_frame_idx + 1, "#" * 10)
        print("anno", self.anno[self.cur_frame_idx])
        print("occ", self.occ[self.cur_frame_idx])
        print("oov", self.oov[self.cur_frame_idx])
        print("annotated frame:",[x+1 for x,y in self.anno.items() if y!=-1])

    def prev_frame(self, event=None):
        self.drawing = False
        self.cur_frame_idx -= 1
        if self.cur_frame_idx < 0:
            messagebox.showinfo("Info", "It's the initial frame!")
            self.cur_frame_idx = 0
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cur_frame_idx)
            if not self.oov_flag:
                self.out_of_view_var.set(False)
            else:
                self.out_of_view_var.set(True)
            if not self.occ_flag:
                self.occlusion_var.set(False)
            else:
                self.occlusion_var.set(True)
        self.show_frame(self.sequences[self.seq_idx])
        os.system("cls")
        print("#" * 10, "frame:", self.cur_frame_idx + 1, "#" * 10)
        print("anno", self.anno[self.cur_frame_idx])
        print("occ", self.occ[self.cur_frame_idx])
        print("oov", self.oov[self.cur_frame_idx])
        print("annotated frame:",[x+1 for x,y in self.anno.items() if y!=-1])
        if not self.cap:
            return


    def saving_all_things(self):
        #print("saving!")
        #os.system("pause")
        Sequences_path = os.path.join(self.path, "sequences")
        Anno_path = os.path.join(self.path, "anno")
        Full_occlusion_path = os.path.join(self.path, "full_occlusion")
        out_of_view = os.path.join(self.path, "out_of_view")
        os.makedirs(Sequences_path, exist_ok=True)
        os.makedirs(Anno_path, exist_ok=True)
        os.makedirs(Full_occlusion_path, exist_ok=True)
        os.makedirs(out_of_view, exist_ok=True)
        self.anno = defaultdict(lambda: -1,sorted(self.anno.items()))
        self.occ = defaultdict(lambda: -1,sorted(self.occ.items()))
        self.oov = defaultdict(lambda: -1,sorted(self.oov.items()))
        print(Anno_path)
        with open(os.path.join(Anno_path, self.sequence_name + ".txt"), "w") as f:
            for k,j in self.anno.items():
                #print(k+1,":",j)
                if j != -1:
                    x, y, w, h = j
                    f.write(f"{x} {y} {w} {h}\n")
        
        with open(os.path.join(Full_occlusion_path, self.sequence_name + "_full_occlusion.txt"), "w") as f:
            tmp_flag = True
            for k in self.occ.values():
                if k != -1:
                    if tmp_flag:
                        f.write(str(k))
                        tmp_flag = False
                    else:
                        f.write(",")
                        f.write(str(k))
            f.write("\n")

        with open(os.path.join(out_of_view, self.sequence_name + "_out_of_view.txt"), "w") as f:
            tmp_flag = True
            for k in self.oov.values():
                if k != -1:
                    if tmp_flag:
                        f.write(str(k))
                        tmp_flag = False
                    else:
                        f.write(",")
                        f.write(str(k))
            f.write("\n")
        tmp_count = 1
        for i, j in self.imgs.items():
            if j is not None and isinstance(j, np.ndarray):
                j = j.astype(np.uint8)
                os.makedirs(os.path.join(Sequences_path, self.sequence_name), exist_ok=True)
                cv2.imwrite(os.path.join(Sequences_path, self.sequence_name, f'img_{tmp_count:05}.jpg'), j)
                tmp_count +=1

        self.anno = defaultdict(lambda: -1)
        self.occ = defaultdict(lambda: -1)
        self.oov = defaultdict(lambda: -1)
        self.imgs = defaultdict(lambda: -1)
        
    def next_frame(self, event=None):
        self.drawing = False
        self.cur_frame_idx += 1
        #print(self.occ.values())
        #os.system("pause")
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.cur_frame_idx >= total_frames:
            response = messagebox.askquestion("Info", "Want to go to next sequence?")
            if response == 'yes':
                self.seq_idx += 1
                #self.tmp_flag = True
                if self.seq_idx < len(self.sequences):
                    self.cur_frame_idx = 0
                    self.cap.release()
                    self.cap = None
                    if 0 in self.occ.values() or 1 in self.occ.values():
                        self.saving_all_things()
                    self.show_frame(self.sequences[self.seq_idx])
                else:
                    self.seq_idx = len(self.sequences) - 1
            else:
                self.cur_frame_idx = total_frames - 1
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cur_frame_idx)
            if not self.oov_flag:
                self.out_of_view_var.set(False)
            else:
                self.out_of_view_var.set(True)
            if not self.occ_flag:
                self.occlusion_var.set(False)
            else:
                self.occlusion_var.set(True)
            self.show_frame(self.sequences[self.seq_idx])
        os.system("cls")
        print("#" * 10, "frame:", self.cur_frame_idx + 1, "#" * 10)
        print("anno", self.anno[self.cur_frame_idx])
        print("occ", self.occ[self.cur_frame_idx])
        print("oov", self.oov[self.cur_frame_idx])
        print("annotated frame:",[x+1 for x,y in self.anno.items() if y!=-1])
        if not self.cap:
            return


    def skip_sequence(self):
        self.seq_idx += 1
        if self.seq_idx < len(self.sequences):
            self.cur_frame_idx = 0
            self.cap.release()
            self.cap = None
            if 0 in self.occ.values() or 1 in self.occ.values():
                self.saving_all_things()
            self.sequence_name, _ = self.sequences[self.seq_idx]
            self.show_frame(self.sequences[self.seq_idx])
            os.system("cls")
        else:
            self.saving_all_things()
            self.seq_idx = len(self.sequences) - 1
            os.system("cls")
            print("#" * 10, "frame:", self.cur_frame_idx + 1, "#" * 10)
            print("anno", self.anno[self.cur_frame_idx])
            print("occ", self.occ[self.cur_frame_idx])
            print("oov", self.oov[self.cur_frame_idx])
            print("annotated frame:",[x+1 for x,y in self.anno.items() if y!=-1])
            result = messagebox.askquestion("Labeling Finished", "The labels have been finished. Do you want to quit the program?")
            if result == 'yes':
                self.root.destroy()
        
    def jump_to_frame(self):
        
        target_frame = int(self.frame_jump_entry.get())
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.remove_entry_focus()
        if 0 <= target_frame-1 < total_frames:
            self.cur_frame_idx = (target_frame-1)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cur_frame_idx)
            self.show_frame(self.sequences[self.seq_idx])
            if not self.oov_flag:
                self.out_of_view_var.set(False)
            else:
                self.out_of_view_var.set(True)
            if not self.occ_flag:
                self.occlusion_var.set(False)
            else:
                self.occlusion_var.set(True)
            os.system("cls")
            print("#" * 10, "frame:", self.cur_frame_idx + 1, "#" * 10)
            print("anno", self.anno[self.cur_frame_idx])
            print("occ", self.occ[self.cur_frame_idx])
            print("oov", self.oov[self.cur_frame_idx])
            print("annotated frame:",[x+1 for x,y in self.anno.items() if y!=-1])

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    root.update()
    app = AnnotationToolApp(root)
    app.run()
