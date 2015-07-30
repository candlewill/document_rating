# coding=utf-8

from multiprocessing import Pool
import requests

nos = []
for i in range(1000000, 1039999, 1):
    nos.append(str(i))


def get_pic(no):
    f = open("./images/s" + no + '.jpg', 'wb')
    try:
        pic = requests.get(
            "https://portalx.yzu.edu.tw/PortalSocialVB/Include/ShowImage.aspx?ShowType=UserPic&UserAccount=s%s&UserPictureName=" % (
                no)).content
        f.write(pic)
    except:
        pass
    f.close()


if __name__ == '__main__':
    pool = Pool()
    pool.map(get_pic, nos)
    pool.close()
    pool.join()
