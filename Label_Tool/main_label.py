#-------------------------------------------------------------------------------
# Name:        Object bounding box label tool
# Purpose:     Label object bboxes for ImageNet Detection data
# Author:      Qiushi
# Created:     06/06/2014

#
#-------------------------------------------------------------------------------
try:
    from Tkinter import *  # Python 2.x
except ImportError:
    from tkinter import *  # Python 3.x

from PIL import Image, ImageTk
import os
import glob
from itertools import combinations
import re

# colors for the bboxes
COLORS = ['red', 'blue', 'yellow', 'pink', 'cyan', 'green', 'black']
# image sizes for the examples
SIZE = 256, 256


# Every image should be labeled by two users, and every pair or users should have the same number of shared images.
# Divide the list of images into users*(users-1)/2 parts. Then, this function will tell which part should be assigned to
# each user. Each entry of the "matrix" is a list of indices that correspond to the parts.
def get_cross_label_matrix(users):
    if users <= 2:
        return [[0], [0]]
    else:
        new_width = users * (users - 1) // 2
        matrix = get_cross_label_matrix(users - 1)
        for i in range(len(matrix)):
            matrix[i].append(new_width - 1 - i)
        matrix.append([i for i in range(new_width - users + 1, new_width)])
        return matrix


class LabelTool():
    def __init__(self, master):
        # set up the main frame
        self.parent = master
        self.parent.title("LabelTool")
        self.frame = Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width = FALSE, height = FALSE)

        # initialize global state
        self.imageDir = ''
        self.imageList = []
        self.diffList = []
        self.ten_percent_outline = None
        self.egDir = ''
        self.egList = []
        self.outDir = ''
        self.cur = 0
        self.total = 0
        self.category = 0
        self.imagename = ''
        self.labelfilename = ''
        self.tkimg = None

        # initialize mouse state
        self.STATE = {}
        self.STATE['click'] = 0
        self.STATE['x'], self.STATE['y'] = 0, 0

        # reference to bbox
        self.bboxIdList = []
        self.bboxId = None
        self.bboxList = []
        self.hl = None
        self.vl = None

        # ----------------- GUI stuff ---------------------
        # dir entry & load
        self.label = Label(self.frame, text = "Image Dir:")
        self.label.grid(row = 0, column = 0, sticky = E)
        self.entry = Entry(self.frame)
        self.entry.grid(row = 0, column = 1, sticky = W+E)
        try:
            self.entry.insert(0, open('lastfolder.txt').readline().strip())
        except FileNotFoundError:
            self.entry.insert(0, '../../images')
        self.listbox_names = sorted(['Emmanuel', 'Jasper', 'Lennart', 'Matthias'])
        self.name_listbox = Listbox(self.frame, selectmode=SINGLE, height=1+len(self.listbox_names))
        self.name_listbox.grid(row = 0, column = 2, sticky = W+E)
        self.name_listbox.insert(END, '-- No User --')
        self.name_listbox.selection_set(0)
        for name in self.listbox_names:
            self.name_listbox.insert(END, name)
        self.ldBtn = Button(self.frame, text = "Load", command = self.loadDir)
        self.ldBtn.grid(row = 0, column = 3, sticky = W+E)

        # main panel for labeling
        self.mainPanel = Canvas(self.frame, cursor='tcross')
        self.mainPanel.bind("<Button-1>", self.mouseClick)
        self.mainPanel.bind("<Motion>", self.mouseMove)
        self.parent.bind("<Escape>", self.cancelBBox)  # press <Espace> to cancel current bbox
        self.parent.bind("s", self.cancelBBox)
        self.parent.bind("<BackSpace>", self.prevImage) # press '<BackSpace>' to go backforward
        self.parent.bind("0", lambda ev: self.nextImage('0', ev)) # press 'd' to go forward
        self.parent.bind("1", lambda ev: self.nextImage('1', ev))
        self.parent.bind("2", lambda ev: self.nextImage('2', ev))
        self.parent.bind("3", lambda ev: self.nextImage('3', ev))
        self.parent.bind("4", lambda ev: self.nextImage('4', ev))
        self.mainPanel.grid(row = 1, column = 1, rowspan = 4, sticky = W+N)

        # showing bbox info & delete bbox
        self.lb1 = Label(self.frame, text = 'Bounding boxes:')
        self.lb1.grid(row = 1, column = 2,  sticky = W+N)
        self.listbox = Listbox(self.frame, width = 22, height = 12)
        self.listbox.grid(row = 2, column = 2, sticky = N)
        self.btnDel = Button(self.frame, text = 'Delete', command = self.delBBox)
        self.btnDel.grid(row = 3, column = 2, sticky = W+E+N)
        self.btnClear = Button(self.frame, text = 'ClearAll', command = self.clearBBox)
        self.btnClear.grid(row = 4, column = 2, sticky = W+E+N)

        # control panel for image navigation
        self.ctrPanel = Frame(self.frame)
        self.ctrPanel.grid(row = 5, column = 1, columnspan = 2, sticky = W+E)
        self.prevBtn = Button(self.ctrPanel, text='<< Prev', width = 10, command = self.prevImage)
        self.prevBtn.pack(side = LEFT, padx = 5, pady = 3)
        self.nextBtnT = Button(self.ctrPanel, text='1: Voltaik', width = 10, command=lambda: self.nextImage('1'))
        self.nextBtnT.pack(side = LEFT, padx = 5, pady = 3)
        self.nextBtnU = Button(self.ctrPanel, text='2: Thermie', width = 10, command=lambda: self.nextImage('2'))
        self.nextBtnU.pack(side = LEFT, padx = 5, pady = 3)
        self.nextBtnB = Button(self.ctrPanel, text='3: Beides', width = 10, command=lambda: self.nextImage('3'))
        self.nextBtnB.pack(side = LEFT, padx = 5, pady = 3)
        self.nextBtnA = Button(self.ctrPanel, text='4: Unsicher', width = 10, command=lambda: self.nextImage('4'))
        self.nextBtnA.pack(side = LEFT, padx = 5, pady = 3)
        self.nextBtnF = Button(self.ctrPanel, text='0: Gar nichts', width = 10, command=lambda: self.nextImage('0'))
        self.nextBtnF.pack(side = LEFT, padx = 5, pady = 3)
        self.progLabel = Label(self.ctrPanel, text = "Progress:     /    ")
        self.progLabel.pack(side = LEFT, padx = 5)
        self.tmpLabel = Label(self.ctrPanel, text = "Go to Image No.")
        self.tmpLabel.pack(side = LEFT, padx = 5)
        self.idxEntry = Entry(self.ctrPanel, width = 5)
        self.idxEntry.pack(side = LEFT)
        self.goBtn = Button(self.ctrPanel, text = 'Go', command = self.gotoImage)
        self.goBtn.pack(side = LEFT)

        # example pannel for illustration
        self.egPanel = Frame(self.frame, border = 10)
        self.egPanel.grid(row = 1, column = 0, rowspan = 5, sticky = N)
        self.tmpLabel2 = Label(self.egPanel, text = "Options:")
        self.tmpLabel2.pack(side = TOP, pady = 5)
        self.show_ten_percent = IntVar()
        self.show_ten_percent_button = Checkbutton(self.egPanel, text='Show 10% outline', variable=self.show_ten_percent)
        self.show_ten_percent_button.select()
        self.show_ten_percent_button.pack(side=TOP)
        self.diff_button = Button(self.egPanel, text="Diff labels*.csv", command=self.label_diff)
        self.diff_button.pack(side=TOP, pady=5)
        self.diff_label_text = StringVar()
        self.diff_label = Label(self.egPanel, textvariable=self.diff_label_text, anchor=E, justify=LEFT)
        self.diff_label.pack(side=TOP)
        self.egLabels = []

        # display mouse position
        self.disp = Label(self.ctrPanel, text='')
        self.disp.pack(side = RIGHT)

        self.frame.columnconfigure(1, weight = 1)
        self.frame.rowconfigure(4, weight = 1)

        self.csv_filename = 'labels.csv'

        # for debugging
##        self.setImage()
##        self.loadDir()


    @staticmethod
    def extract_username(filename):
        username = re.search(r'labels(\w*)\.csv', filename)
        if username:
            return username.group(1)
        else:
            return ''

    def label_diff(self):
        csv_filelist = glob.glob('labels*.csv')
        print('Running diff over:')
        print(csv_filelist)
        self.imageDir = self.entry.get().strip()
        self.category = self.imageDir
        self.imageList = []
        self.diffList = []
        self.csv_filename = 'label_diff.csv'
        if os.path.isfile(self.csv_filename):
            backup_index = 0
            while True:
                backup_index += 1
                target_name = 'label_diff' + str(backup_index) + '.csv'
                # If the name doesn't exist, break the loop
                if not os.path.isfile(target_name):
                    break
            os.rename(self.csv_filename, target_name)
            print('Renamed existing diff file to ' + target_name)
        csv_file = open(self.csv_filename, 'w')
        for file1, file2 in combinations(csv_filelist, 2):
            username1 = self.extract_username(file1)
            username2 = self.extract_username(file2)
            filelist1 = [f.strip().split(',') for f in open(file1).readlines()]
            filelist2 = [f.strip().split(',') for f in open(file2).readlines()]
            common_files = set([f[0] for f in filelist1]) & set([f[0] for f in filelist2])
            filelist1 = {f[0]: f[1] for f in filelist1 if f[0] in common_files}
            filelist2 = {f[0]: f[1] for f in filelist2 if f[0] in common_files}
            common_files = sorted(common_files)
            for file in common_files:
                if filelist1[file] != filelist2[file]:
                    self.imageList.append(os.path.join(self.imageDir, file))
                    self.diffList.append('{}: {}\n{}: {}'.format(username1, filelist1[file], username2, filelist2[file]))
                else:
                    csv_file.write(file + ',' + filelist1[file] + '\n')
        csv_file.close()

        if not self.imageList:
            self.diff_label_text.set('No differences found')
            print(self.diff_label_text.get())
        else:
            self.loadDir(diff=True)

    def loadDir(self, dbg = False, diff=False):
        self.diff_label_text.set('')
        if not diff:
            if not dbg:
                s = self.entry.get()
                self.parent.focus()
                self.category = s.strip()
            else:
                s = r'D:\workspace\python\labelGUI'
    ##        if not os.path.isdir(s):
    ##            tkMessageBox.showerror("Error!", message = "The specified dir doesn't exist!")
    ##            return
            # get image list
            self.imageDir = self.category
            self.imageList = sorted(glob.glob(os.path.join(self.imageDir, '*.png')))

            self.diffList = []

            if self.name_listbox.curselection():
                user_filter = self.name_listbox.curselection()[0]
            else:
                user_filter = 0
            if type(user_filter) != int:
                user_filter = int(user_filter)

            if user_filter > 0:
                self.csv_filename = 'labels' + self.listbox_names[user_filter - 1] + '.csv'
                user_count = len(self.listbox_names)
                fullList = self.imageList
                parts = user_count * (user_count-1) // 2
                imageListSteps = [int(float(len(fullList)) / parts * s) for s in range(0, parts + 1)]

                self.imageList = []
                for i in get_cross_label_matrix(user_count)[user_filter - 1]:
                    self.imageList.extend(fullList[imageListSteps[i]:imageListSteps[i+1]])
                del fullList
            else:
                self.csv_filename = 'labels.csv'

            if len(self.imageList) == 0:
                print('No .png images found in the specified dir!')
                return

        open('lastfolder.txt', 'w').write(self.imageDir)

        # default to the 1st image in the collection
        self.cur = 1
        if not diff:
            try:
                self.cur = len([line for line in open(self.csv_filename).readlines() if line.strip()]) + 1
                print('Existing labels found, continue from image ' + str(self.cur))
            except FileNotFoundError:
                pass
        self.total = len(self.imageList)

        self.loadImage()
        print('%d images loaded from %s' %(self.total, self.imageDir))

    def loadImage(self):
        # load image
        imagepath = self.imageList[self.cur - 1]
        self.img = Image.open(imagepath)
        self.tkimg = ImageTk.PhotoImage(self.img)
        self.mainPanel.config(width = max(self.tkimg.width(), 400), height = max(self.tkimg.height(), 400))
        self.mainPanel.create_image(0, 0, image = self.tkimg, anchor=NW)
        self.progLabel.config(text = "%04d/%04d" %(self.cur, self.total))

        # add/remove 10% outline
        if self.ten_percent_outline:
            self.mainPanel.delete(self.ten_percent_outline)
            self.ten_percent_outline = None

        if self.show_ten_percent.get():
            self.ten_percent_outline = self.mainPanel.create_rectangle(
                self.img.width // 10, self.img.height // 10,
                self.img.width - (self.img.width // 10), self.img.height - (self.img.height // 10),
                width=1,
                outline='pink'
            )

        if self.diffList:
            self.diff_label_text.set(self.diffList[self.cur - 1])

        # load labels
        self.clearBBox()
        self.imagename = os.path.split(imagepath)[-1].split('.')[0]
        labelname = self.imagename + '.txt'
        self.labelfilename = os.path.join(self.outDir, labelname)
        bbox_cnt = 0
        if os.path.exists(self.labelfilename):
            with open(self.labelfilename) as f:
                for (i, line) in enumerate(f):
                    if i == 0:
                        bbox_cnt = int(line.strip())
                        continue
                    tmp = [int(t.strip()) for t in line.split()]
##                    print tmp
                    self.bboxList.append(tuple(tmp))
                    tmpId = self.mainPanel.create_rectangle(tmp[0], tmp[1], \
                                                            tmp[2], tmp[3], \
                                                            width = 2, \
                                                            outline = COLORS[(len(self.bboxList)-1) % len(COLORS)])
                    self.bboxIdList.append(tmpId)
                    self.listbox.insert(END, '(%d, %d) -> (%d, %d)' %(tmp[0], tmp[1], tmp[2], tmp[3]))
                    self.listbox.itemconfig(len(self.bboxIdList) - 1, fg = COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])

    def saveImage(self):
        with open(self.labelfilename, 'w') as f:
            f.write('%d\n' %len(self.bboxList))
            for bbox in self.bboxList:
                f.write(' '.join(map(str, bbox)) + '\n')
        print('Image No. %d saved' %(self.cur))


    def mouseClick(self, event):
        if self.STATE['click'] == 0:
            self.STATE['x'], self.STATE['y'] = event.x, event.y
        else:
            x1, x2 = min(self.STATE['x'], event.x), max(self.STATE['x'], event.x)
            y1, y2 = min(self.STATE['y'], event.y), max(self.STATE['y'], event.y)
            self.bboxList.append((x1, y1, x2, y2))
            self.bboxIdList.append(self.bboxId)
            self.bboxId = None
            self.listbox.insert(END, '(%d, %d) -> (%d, %d)' %(x1, y1, x2, y2))
            self.listbox.itemconfig(len(self.bboxIdList) - 1, fg = COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])
        self.STATE['click'] = 1 - self.STATE['click']

    def mouseMove(self, event):
        self.disp.config(text = 'x: %d, y: %d' %(event.x, event.y))
        if self.tkimg:
            if self.hl:
                self.mainPanel.delete(self.hl)
            self.hl = self.mainPanel.create_line(0, event.y, self.tkimg.width(), event.y, width = 2)
            if self.vl:
                self.mainPanel.delete(self.vl)
            self.vl = self.mainPanel.create_line(event.x, 0, event.x, self.tkimg.height(), width = 2)
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
            self.bboxId = self.mainPanel.create_rectangle(self.STATE['x'], self.STATE['y'], \
                                                            event.x, event.y, \
                                                            width = 2, \
                                                            outline = COLORS[len(self.bboxList) % len(COLORS)])

    def cancelBBox(self, event):
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
                self.bboxId = None
                self.STATE['click'] = 0

    def delBBox(self):
        sel = self.listbox.curselection()
        if len(sel) != 1 :
            return
        idx = int(sel[0])
        self.mainPanel.delete(self.bboxIdList[idx])
        self.bboxIdList.pop(idx)
        self.bboxList.pop(idx)
        self.listbox.delete(idx)

    def clearBBox(self):
        for idx in range(len(self.bboxIdList)):
            self.mainPanel.delete(self.bboxIdList[idx])
        self.listbox.delete(0, len(self.bboxList))
        self.bboxIdList = []
        self.bboxList = []

    def prevImage(self, event = None):
        #self.saveImage()
        df = [line for line in open(self.csv_filename).readlines() if line.strip()]
        open(self.csv_filename, 'w').writelines(df[:-1])
        if self.cur > 1:
            self.cur -= 1
            self.loadImage()

    def nextImage(self, label, event=None):
        with open(self.csv_filename, 'a') as csvfile:
            csvfile.write(','.join([os.path.basename(self.imageList[self.cur-1]), label]) + '\n')
        if self.cur < self.total:
            self.cur += 1
            self.loadImage()

    def gotoImage(self):
        idx = int(self.idxEntry.get())
        if 1 <= idx and idx <= self.total:
            self.saveImage()
            self.cur = idx
            self.loadImage()

##    def setImage(self, imagepath = r'test2.png'):
##        self.img = Image.open(imagepath)
##        self.tkimg = ImageTk.PhotoImage(self.img)
##        self.mainPanel.config(width = self.tkimg.width())
##        self.mainPanel.config(height = self.tkimg.height())
##        self.mainPanel.create_image(0, 0, image = self.tkimg, anchor=NW)

if __name__ == '__main__':
    root = Tk()
    tool = LabelTool(root)
    root.resizable(width =  True, height = True)
    root.mainloop()
