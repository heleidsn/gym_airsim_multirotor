'''
LGMD perception method
'''

class LGMD():
    def __init__(self) -> None:
        self.img_pre = None
        self.img_cur = None
        self.lgmd_out = None

    def reset(self):
        self.img_pre = None
        self.img_cur = None

    def update(self, rgb_img):
        if self.img_pre == None:
            self.img_pre = rgb_img
        else:
            self.img_cur = rgb_img
            self.calculate_lgmd()

        return self.lgmd_out

    def calculate_lgmd(self):
        self.lgmd_out = self.img_cur - self.img_pre