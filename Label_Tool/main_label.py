# -------------------------------------------------------------------------------
# Name:        Image category label tool
# Author:      Original bbox tool by Qiushi; category modifications by Matthias and Lennart
# Created:     06/06/2014

# Updated:     Changed functionality to categories February 2018
# -------------------------------------------------------------------------------
try:
    from Tkinter import *  # Python 2.x
except ImportError:
    from tkinter import *  # Python 3.x

from PIL import Image, ImageTk
import os
import glob
from itertools import combinations
import re


class LabelTool:
    def __init__(self, master):
        # set up the main frame
        self.parent = master
        self.parent.title("LabelTool")
        self.frame = Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width=FALSE, height=FALSE)

        # initialize global state
        self.image_dir = ''
        self.image_list = []
        self.diff_list = []
        self.ten_percent_outline = None
        self.current_image = 0
        self.total = 0
        self.image_name = ''
        self.label_file_name = ''
        self.img = None
        self.tkimg = None
        self.diff_base_filename = 'label_diff{}.csv'
        self.lastfolder_filename = 'lastfolder.txt'

        # ----------------- GUI stuff ---------------------
        # dir entry & load
        self.label = Label(self.frame, text="Image Dir:")
        self.label.grid(row=0, column=0, sticky=E)
        self.entry = Entry(self.frame)
        self.entry.grid(row=0, column=1, sticky=W + E)
        try:
            self.entry.insert(0, open(self.lastfolder_filename).readline().strip())
        except FileNotFoundError:
            self.entry.insert(0, '../../images')

        self.load_button = Button(self.frame, text="Load", command=self.load_directory)
        self.load_button.grid(row=0, column=3, sticky=W + E)

        # main panel for labeling
        self.main_panel = Canvas(self.frame, cursor='tcross')
        # Need dummy parameter _ for unused event
        self.parent.bind("<BackSpace>", lambda _: self.previous_image())  # press '<BackSpace>' to go backforward
        # Keys 0 to 4 for the corresponding categories
        self.parent.bind("0", lambda _: self.next_image('0'))
        self.parent.bind("1", lambda _: self.next_image('1'))
        self.parent.bind("2", lambda _: self.next_image('2'))
        self.parent.bind("3", lambda _: self.next_image('3'))
        self.parent.bind("4", lambda _: self.next_image('4'))
        self.main_panel.grid(row=1, column=1, rowspan=4, sticky=W + N)

        # showing users
        self.lb1 = Label(self.frame, text='Users:')
        self.lb1.grid(row=1, column=2, sticky=W + N)
        self.listbox_names = sorted(['Emmanuel', 'Jasper', 'Lennart', 'Matthias'])
        self.name_listbox = Listbox(self.frame, selectmode=SINGLE, height=1 + len(self.listbox_names))
        self.name_listbox.grid(row=2, column=2, sticky=W + E)
        self.name_listbox.insert(END, '-- No User --')
        self.name_listbox.selection_set(0)
        for name in self.listbox_names:
            self.name_listbox.insert(END, name)

        # control panel for image navigation
        self.control_panel = Frame(self.frame)
        self.control_panel.grid(row=5, column=1, columnspan=2, sticky=W + E)
        self.previous_button = Button(self.control_panel, text='<< Prev', width=10, command=self.previous_image)
        self.previous_button.pack(side=LEFT, padx=5, pady=3)

        self.next_button_category1 = Button(self.control_panel, text='1: Voltaik',
                                            width=10, command=lambda: self.next_image('1'))
        self.next_button_category1.pack(side=LEFT, padx=5, pady=3)

        self.next_button_category2 = Button(self.control_panel, text='2: Thermie',
                                            width=10, command=lambda: self.next_image('2'))
        self.next_button_category2.pack(side=LEFT, padx=5, pady=3)

        self.next_button_category3 = Button(self.control_panel, text='3: Beides',
                                            width=10, command=lambda: self.next_image('3'))
        self.next_button_category3.pack(side=LEFT, padx=5, pady=3)

        self.next_button_category4 = Button(self.control_panel, text='4: Unsicher',
                                            width=10, command=lambda: self.next_image('4'))
        self.next_button_category4.pack(side=LEFT, padx=5, pady=3)

        self.next_button_category0 = Button(self.control_panel, text='0: Gar nichts',
                                            width=10, command=lambda: self.next_image('0'))
        self.next_button_category0.pack(side=LEFT, padx=5, pady=3)

        self.progress_label = Label(self.control_panel, text="Progress:     /    ")
        self.progress_label.pack(side=LEFT, padx=5)
        self.go_to_image_label = Label(self.control_panel, text="Go to Image No.")
        self.go_to_image_label.pack(side=LEFT, padx=5)
        self.go_to_image_entry = Entry(self.control_panel, width=5)
        self.go_to_image_entry.pack(side=LEFT)
        self.go_to_image_button = Button(self.control_panel, text='Go', command=self.go_to_image)
        self.go_to_image_button.pack(side=LEFT)

        # Options panel
        self.options_panel = Frame(self.frame, border=10)
        self.options_panel.grid(row=1, column=0, rowspan=5, sticky=N)
        self.options_panel_title = Label(self.options_panel, text="Options:")
        self.options_panel_title.pack(side=TOP, pady=5)
        self.show_ten_percent = IntVar()
        self.show_ten_percent_button = Checkbutton(self.options_panel, text='Show 10% outline',
                                                   variable=self.show_ten_percent)
        self.show_ten_percent_button.select()
        self.show_ten_percent_button.pack(side=TOP)
        self.diff_button = Button(self.options_panel, text="Diff labels*.csv", command=self.label_diff)
        self.diff_button.pack(side=TOP, pady=5)
        self.diff_label_text = StringVar()
        self.diff_label = Label(self.options_panel, textvariable=self.diff_label_text, anchor=E, justify=LEFT)
        self.diff_label.pack(side=TOP)

        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(4, weight=1)

        self.csv_filename = 'labels.csv'

    @staticmethod
    def extract_username(filename):
        username = re.search(r'labels(\w*)\.csv', filename)
        if username:
            return username.group(1)
        else:
            return ''

    def diff_to_file(self):
        csv_filename = self.diff_base_filename.format('')
        if not os.path.isfile(csv_filename):
            return
        filelist_diff = [f.strip().split(',') for f in open(csv_filename).readlines() if f.strip()]
        filelist_diff = {f[0]: f[1] for f in filelist_diff}
        csv_filelist = glob.glob('labels*.csv')
        for file1 in csv_filelist:
            filelist1 = [f.strip().split(',') for f in open(file1).readlines() if f.strip()]
            upd_file = open(file1, 'w')
            for label in filelist1:
                if label[0] in filelist_diff:
                    label[1] = filelist_diff[label[0]]
                upd_file.write(label[0] + ',' + label[1] + '\n')
            upd_file.close()

    @staticmethod
    # Every image should be labeled by two users, and every pair or users should have the same number of shared images.
    # Divide the list of images into users*(users-1)/2 parts. Then, this function will tell which part should be
    # assigned to each user. Each entry of the "matrix" is a list of indices that correspond to the parts.
    def get_cross_label_matrix(users):
        if users <= 2:
            return [[0], [0]]
        else:
            new_width = users * (users - 1) // 2
            matrix = LabelTool.get_cross_label_matrix(users - 1)
            for i in range(len(matrix)):
                matrix[i].append(new_width - 1 - i)
            matrix.append([i for i in range(new_width - users + 1, new_width)])
            return matrix

    def label_diff(self):
        csv_filelist = glob.glob('labels*.csv')
        print('Running diff over:')
        print(csv_filelist)
        if len(csv_filelist) < 2:
            self.diff_label_text.set('Need 2 or more\nCSV files for diff')
            return
        self.image_dir = self.entry.get().strip()
        self.image_list = []
        self.diff_list = []
        self.csv_filename = self.diff_base_filename.format('')
        if os.path.isfile(self.csv_filename):
            backup_index = 0
            while True:
                backup_index += 1
                target_name = self.diff_base_filename.format(backup_index)
                # If the name doesn't exist, break the loop
                if not os.path.isfile(target_name):
                    break
            os.rename(self.csv_filename, target_name)
            print('Renamed existing diff file to ' + target_name)
        csv_file = open(self.csv_filename, 'w')
        for file1, file2 in combinations(csv_filelist, 2):
            username1 = LabelTool.extract_username(file1)
            username2 = LabelTool.extract_username(file2)
            filelist1 = [f.strip().split(',') for f in open(file1).readlines()]
            filelist2 = [f.strip().split(',') for f in open(file2).readlines()]
            common_files = set([f[0] for f in filelist1]) & set([f[0] for f in filelist2])
            filelist1 = {f[0]: f[1] for f in filelist1 if f[0] in common_files}
            filelist2 = {f[0]: f[1] for f in filelist2 if f[0] in common_files}
            common_files = sorted(common_files)
            different_label_counter = 0
            for file in common_files:
                if filelist1[file] != filelist2[file]:
                    self.image_list.append(os.path.join(self.image_dir, file))
                    self.diff_list.append('{}: {}\n{}: {}'.format(username1, filelist1[file],
                                                                  username2, filelist2[file]))
                    different_label_counter += 1
                else:
                    csv_file.write(file + ',' + filelist1[file] + '\n')
            print('{} different labels (out of {}, {}%) for {} and {}'.format(
                different_label_counter,
                len(common_files),
                round(float(different_label_counter) / len(common_files) * 100., 1),
                username1,
                username2
            ))
        csv_file.close()

        if not self.image_list:
            self.diff_label_text.set('No differences found')
            print(self.diff_label_text.get())
        else:
            self.load_directory(diff=True)

    def load_directory(self, diff=False):
        self.diff_label_text.set('')
        if not diff:
            s = self.entry.get()
            self.parent.focus()
            self.image_dir = s.strip()

            # get image list
            self.image_list = sorted(glob.glob(os.path.join(self.image_dir, '*.png')))

            # reset diff_list to be empty
            self.diff_list = []

            if self.name_listbox.curselection():
                user_filter = self.name_listbox.curselection()[0]
            else:
                user_filter = 0
            if type(user_filter) != int:
                user_filter = int(user_filter)

            if user_filter > 0:
                self.csv_filename = 'labels' + self.listbox_names[user_filter - 1] + '.csv'
                user_count = len(self.listbox_names)
                full_list = self.image_list
                parts = user_count * (user_count - 1) // 2
                image_list_steps = [int(float(len(full_list)) / parts * s) for s in range(0, parts + 1)]

                self.image_list = []
                for i in LabelTool.get_cross_label_matrix(user_count)[user_filter - 1]:
                    self.image_list.extend(full_list[image_list_steps[i]:image_list_steps[i + 1]])
                del full_list
            else:
                self.csv_filename = 'labels.csv'

            if len(self.image_list) == 0:
                print('No .png images found in the specified dir!')
                return

        open(self.lastfolder_filename, 'w').write(self.image_dir)

        # default to the 1st image in the collection
        self.current_image = 1
        self.total = len(self.image_list)
        if not diff:
            try:
                self.current_image = len([line for line in open(self.csv_filename).readlines() if line.strip()]) + 1
                self.current_image = min(self.current_image, self.total)
                print('Existing labels found, continue from image ' + str(self.current_image))
            except FileNotFoundError:
                pass

        self.load_image()
        print('%d images loaded from %s' % (self.total, self.image_dir))

    def load_image(self):
        # load image
        imagepath = self.image_list[self.current_image - 1]
        self.img = Image.open(imagepath)
        self.tkimg = ImageTk.PhotoImage(self.img)
        self.main_panel.config(width=max(self.tkimg.width(), 400), height=max(self.tkimg.height(), 400))
        self.main_panel.create_image(0, 0, image=self.tkimg, anchor=NW)
        self.progress_label.config(text="%04d/%04d" % (self.current_image, self.total))

        # add/remove 10% outline
        if self.ten_percent_outline:
            self.main_panel.delete(self.ten_percent_outline)
            self.ten_percent_outline = None

        if self.show_ten_percent.get():
            self.ten_percent_outline = self.main_panel.create_rectangle(
                self.img.width // 10, self.img.height // 10,
                self.img.width - (self.img.width // 10), self.img.height - (self.img.height // 10),
                width=1,
                outline='pink'
            )

        if self.diff_list:
            self.diff_label_text.set(self.diff_list[self.current_image - 1])

    def previous_image(self):
        df = [line for line in open(self.csv_filename).readlines() if line.strip()]
        open(self.csv_filename, 'w').writelines(df[:-1])
        if self.current_image > 1:
            self.current_image -= 1
            self.load_image()

    def next_image(self, label):
        with open(self.csv_filename, 'a') as csv_file:
            csv_file.write(','.join([os.path.basename(self.image_list[self.current_image - 1]), label]) + '\n')
        if self.current_image < self.total:
            self.current_image += 1
            self.load_image()

    def go_to_image(self):
        new_image_id = int(self.go_to_image_entry.get())
        if 1 <= new_image_id <= self.total:
            self.current_image = new_image_id
            self.load_image()


if __name__ == '__main__':
    root = Tk()
    tool = LabelTool(root)
    root.resizable(width=True, height=True)
    root.mainloop()
