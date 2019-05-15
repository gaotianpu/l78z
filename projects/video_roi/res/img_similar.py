from skimage.measure import compare_ssim
import cv2
import os
import sys 


def compare_image(grayA, path_image2):
    '''比较2张图片的相似度'''
    imageB = cv2.imread(path_image2)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    (score, diff) = compare_ssim(grayA, grayB, full=True)
    # print("SSIM: {}".format(score))
    return score

# 文件名： id.mp4
#
# 人工选取命名： id_0.png   id, area_no
# 自动裁剪命名:  id_0_0.png  id, area_no, frame_index

def run(media_id, area_no_list):
    for area_no in area_no_list:
        manual_img = "%s_%s.png" % (media_id,area_no)  
        # print(manual_img)
        imageA = cv2.imread(manual_img)
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)

        i = 0
        while True:
            img_name = "%s_%s_%s.png" % (media_id,area_no,i*125)
            # print(manual_img,img_name)
            if not os.path.exists(img_name):
                break

            score = compare_image(grayA, img_name)
            # if score >= 0.41 : # area_no == '0' or (area_no in ['1', '2'] and score >= 0.55):
            print(area_no, img_name, score)

            i = i + 1


if __name__ == "__main__":
    video_id = sys.argv[1]
    str_area = sys.argv[2]

    run(video_id, str_area.split(',') )
