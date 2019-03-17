import numpy as np
import math
import tesserocr
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from scipy.signal import convolve as conv1d
from sklearn.cluster import KMeans
from formrecognition.recogntion import ocr_grid_recognition

import formrecognition.predict as pred

"""
define some hyper parameters

"""
SINGLE_GRID_WIDTH=10
SINGLE_GRID_HEIGHT=7


class Grid:
    def __init__(self):
        self.zuoxiajiao=None
        self.youxiajiao=None
        self.zuoshangjiao=None
        self.youshangjiao=None
        self.width=None
        self.height=None
        self.text_contents=None
        self.img_array=None
        self.center=None

    def caculate_center(self):
        if (not self.zuoxiajiao==None) and (not self.youxiajiao==None) and (not self.zuoshangjiao==None):
            x,y = (self.zuoshangjiao.x+self.youxiajiao.x)/2 , (self.zuoshangjiao.y+self.youxiajiao.y)/2
            self.center=Feature_point(x,y)


###########################################################################################
"""
判断两个线段间是否存在交点
"""
class point():  # 定义类
    def __init__(self, x, y):
        self.x = x
        self.y = y


def get_distance(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def get_distance_feapoint(p1,p2):
    return math.sqrt((p1.x-p2.x)**2+(p1.y-p2.y)**2)

def cross(p1, p2, p3):  # 跨立实验
    x1 = p2.x - p1.x
    y1 = p2.y - p1.y
    x2 = p3.x - p1.x
    y2 = p3.y - p1.y
    return x1 * y2 - x2 * y1

def IsIntersec(line1, line2):  # 判断两线段是否相交

    p1 = point(line1[0][0], line1[0][1])
    p2 = point(line1[1][0], line1[1][1])
    p3 = point(line2[0][0], line2[0][1])
    p4 = point(line2[1][0], line2[1][1])
    # 快速排斥，以l1、l2为对角线的矩形必相交，否则两线段不相交
    if (max(p1.x, p2.x) >= min(p3.x, p4.x)  # 矩形1最右端大于矩形2最左端
            and max(p3.x, p4.x) >= min(p1.x, p2.x)  # 矩形2最右端大于矩形最左端
            and max(p1.y, p2.y) >= min(p3.y, p4.y)  # 矩形1最高端大于矩形最低端
            and max(p3.y, p4.y) >= min(p1.y, p2.y)):  # 矩形2最高端大于矩形最低端

        # 若通过快速排斥则进行跨立实验
        if (cross(p1, p2, p3) * cross(p1, p2, p4) <= 0
                and cross(p3, p4, p1) * cross(p3, p4, p2) <= 0):
            D = 1
        else:
            D = 0
    else:
        D = 0
    return D

###########################################################################################

def get_line_param(line):
    """
    计算一些参数，后面用于求取相交线段的交点
    :param line:
    :return:
    """
    a=line[0][1]-line[1][1]
    b=line[1][0]-line[0][0]
    c=line[0][0]*line[1][1]-line[1][0]*line[0][1]
    return a,b,c


class Feature_point:
    """
    用于保存交点的特殊点 类
    """
    def __init__(self,x,y):
        self.x=int(x)
        self.y=int(y)
        self.hori_intersect_line=None
        self.verti_intersect_line=None
        self.type=None



# def feature_point_type(point,)

def get_cross_point(hori_line1,verti_line2,shape):
    """
    求取两条直线间的交点

    """
    (x_length,y_length)=shape
    assert hori_line1[0][0]==hori_line1[1][0],"求交点时水平直线出现了某些问题，起点和终点不在一条水平线上"
    assert verti_line2[0][1]==verti_line2[1][1],"求交点时竖直直线出现了某些问题，起点和终点不在一条竖直直线上"

    if hori_line1[0][1]-6>=0 and hori_line1[1][1]+6<=y_length-1:
        hori_line1[0]=(hori_line1[0][0],hori_line1[0][1] - 6)
        hori_line1[1]=(hori_line1[1][0],hori_line1[1][1]+6)
       # hori_line1[0][1] = hori_line1[0][1] - 6
    elif hori_line1[0][1]-6< 0 and hori_line1[1][1]+6<=y_length-1:
        hori_line1[0]=(hori_line1[0][0],0)
        hori_line1[1] = (hori_line1[1][0], hori_line1[1][1] + 6)
    elif hori_line1[0][1]-6>=0 and hori_line1[1][1]+6>y_length-1:
        hori_line1[0] = (hori_line1[0][0], hori_line1[0][1] - 6)
        hori_line1[1]= (hori_line1[1][0],y_length-1)
    else:
        hori_line1[0] = (hori_line1[0][0], 0)
        hori_line1[1] = (hori_line1[1][0], y_length - 1)



    if verti_line2[0][0]-6>=0 and verti_line2[1][0]+6<=x_length-1:
        verti_line2[0]=(verti_line2[0][0]-6,verti_line2[0][1])
        verti_line2[1]=(verti_line2[1][0]+6,verti_line2[1][1])
    elif verti_line2[0][0]-6 >=0 and verti_line2[1][0]+6 >x_length-1:
        verti_line2[0] = (verti_line2[0][0] - 6, verti_line2[0][1])
        verti_line2[1] = (x_length-1 , verti_line2[1][1])
    elif verti_line2[0][0]-6 <0 and verti_line2[1][0]+6<=x_length-1:
        verti_line2[0] = (0 , verti_line2[0][1])
        verti_line2[1] = (verti_line2[1][0] + 6, verti_line2[1][1])
    else:
        verti_line2[0] = (0, verti_line2[0][1])
        verti_line2[1] = (x_length-1, verti_line2[1][1])

    # if verti_line2[1][0]+6<=x_length-1:
    #     verti_line2[1][0]=verti_line2[1][0]+6
    # else:
    #     verti_line2[1][0]=x_length-1
    # 两条直线均延长8个像素，以免出现端点处无法相交的情况


    if IsIntersec(hori_line1,verti_line2):
        line1_a,line1_b,line1_c=get_line_param(hori_line1)
        line2_a,line2_b,line2_c=get_line_param(verti_line2)
        # GetLinePara(l1);
        # GetLinePara(l2);
        d = line1_a * line2_b - line2_a * line1_b

        temp_x = (line1_b * line2_c - line2_b * line1_c) * 1.0 / d
        temp_y = (line1_c * line2_a - line2_c * line1_a) * 1.0 / d
        p=Feature_point(temp_x,temp_y)
        p.hori_intersect_line=hori_line1
        p.verti_intersect_line=verti_line2

        determain_feature_point_type(p)

        return p
    else:

        return None
# def get_feature_point(hori_lines,verti_lines):
#     fea_poi_sets=[]
#
#     for h_l in hori_lines:
#         for v_l in verti_lines:
#             if IsIntersec(h_l,v_l):
#                 temp_point=get_cross_point(h_l,v_l)
#                 fea_poi_sets.append(temp_point)
#
#
#
#   print()


def classfiy_point_type(feature_point,hori_line_mat,verti_line_mat):
    kernal0_h=np.array([[0,0,0,0,0],[0,0,0,0,0],[-100,-100,-100,-100,-100],[0,0,0,0,0],[0,0,0,0,0]])
    kernal0_v=np.array([[0,0,-100,0,0],[0,0,-100,0,0],[0,0,-100,0,0],[0,0,-100,0,0],[0,0,-100,0,0]])
    kernal1=np.array()



def determain_feature_point_type(feature_point):

    p=(feature_point.x,feature_point.y)
    # assert feature_point.hori_intersect_line[0][0]==feature_point.hori_intersect_line[1][0],"确定特征点的类型时发现水平线存在某些错误"
    # assert feature_point.verti_intersect_line[0][1]==feature_point.verti_intersect_line[1][1],"确定特征点的类型时发现竖直线存在某些错误"



    # hori_points=[(feature_point.hori_intersect_line[0][0],ite_y) for ite_y in range(feature_point.hori_intersect_line[0][1],feature_point.hori_intersect_line[1][1]+1)]
    #
    # verti_points=[(ite_x,feature_point.verti_intersect_line[0][1]) for ite_x in range(feature_point.verti_intersect_line[0][0],feature_point.verti_intersect_line[1][0]+1)]

    # print()
    # hori_points

    hori_points=feature_point.hori_intersect_line
    verti_points=feature_point.verti_intersect_line

    assert p in hori_points, "发现当前检测到的特征点不在当前水平直线上P:%s-----\nhori_lines:%s"%(str(p),str(hori_points))
    assert p in verti_points, "发现当前检测得到的特征点不在当前竖直直线上P:%s-----\nvertilines:%s"%(str(p),str(verti_points))

    # if (p not in hori_points) or (p not in verti_points):
    #     feature_point.type='0'
    # kearnal1=


    is_left,is_up,is_right,is_bottom=False,False,False,False

    h_idex=hori_points.index(p)
    v_index=verti_points.index(p)

    if hori_points.index(p)<20:
        is_left=True
    else:
        is_left=False

    # if
    # width,height=hori_line_mat.shape
    #
    #
    # if not (hori_line_mat[p[0],p[1]]==100 and verti_line_mat[p[0],p[1]]==100):
    #     raise TypeError("Not a cross point!")
    #
    # if p[0]+1 < height and p[0]-1>=0:
    #     if (p[1]-10>=0 and (hori_line_mat[p[0],p[1]-10]==0 or hori_line_mat[p[0]-1,p[1]-10] or hori_line_mat[p[0]+1,p[1]+10])):
    #         is_left=True
    #     elif p[1]-10<0:
    #         is_left=True





    if hori_points.index(p)>len(hori_points)-21:
        is_right=True
    else:
        is_right=False

    if verti_points.index(p)<20:
        is_up=True
    else:
        is_up=False

    if verti_points.index(p)>len(verti_points)-21:
        is_bottom=True
    else:
        is_bottom=False

    #
    # if (is_left,is_up,is_right,is_bottom)==(True,False,False,True):
    #     feature_point.type='1'
    # elif(is_left,is_up,is_right,is_bottom)==(False,False,False,True):
    #     feature_point.type='2'
    # elif (is_left,is_up,is_right,is_bottom)==(False,False,True,True):
    #     feature_point.type='3'
    # elif (is_left,is_up,is_right,is_bottom)==(True,False,False,False):
    #     feature_point.type='4'
    # elif (is_left,is_up,is_right,is_bottom)==(False,False,False,False):
    #     feature_point.type='5'
    # elif (is_left,is_up,is_right,is_bottom)==(False,False,True,False):
    #     feature_point.type='6'
    # elif (is_left,is_up,is_right,is_bottom)==(True,True,False,False):
    #     feature_point.type='7'
    # elif (is_left,is_up,is_right,is_bottom)==(False,True,False,False):
    #     feature_point.type='8'
    # elif (is_left,is_up,is_right,is_bottom)==(False,True,True,False):
    #     feature_point.type='9'
    # else:
    #     feature_point.type='unknown'

    #
    if (is_left,is_up,is_right,is_bottom)==(True,True,False,False):
        feature_point.type='1'
    elif(is_left,is_up,is_right,is_bottom)==(False,True,False,False):
        feature_point.type='2'
    elif (is_left,is_up,is_right,is_bottom)==(False,True,True,False):
        feature_point.type='3'
    elif (is_left,is_up,is_right,is_bottom)==(True,False,False,False):
        feature_point.type='4'
    elif (is_left,is_up,is_right,is_bottom)==(False,False,False,False):
        feature_point.type='5'
    elif (is_left,is_up,is_right,is_bottom)==(False,False,True,False):
        feature_point.type='6'
    elif (is_left,is_up,is_right,is_bottom)==(True,False,False,True):
        feature_point.type='7'
    elif (is_left,is_up,is_right,is_bottom)==(False,False,False,True):
        feature_point.type='8'
    elif (is_left,is_up,is_right,is_bottom)==(False,False,True,True):
        feature_point.type='9'
    else:
        feature_point.type='unknown'





def search_grid(feature_point_sets,table_area):
    print("Starting searching grids!")
    #
    # def search_right_bottom_point(left_bottom_point,feature_point_sets):
    #     for ite_poi in feature_point_sets:
    #         if abs(left_bottom_point.x - ite_poi.x)<=2 and left_bottom_point.y < ite_poi.y and ite_poi.y-left_bottom_point.y > 5:
    # session, predict_op, feed_dict=pred.restore_tf_model()

    grid_list=[]



    for i in range(len(feature_point_sets)):
        left_down_point = ['4', '5', '7', '8']
        temp_grid=Grid()
        # p_left_down=None
        # p_right_down=None
        # p_left_up=None

        if feature_point_sets[i].type in left_down_point:

            def find_nearest(left_down,candi_lup):
                near=None

                thre=10000
                for c in candi_lup:
                    dis=abs(left_down.x-c.x)+abs(left_down.y-c.y)
                    if dis<thre:
                        near=c
                        thre=dis
                return near


            # p_left_down=fea_poi
            temp_grid.zuoxiajiao=feature_point_sets[i]
            right_down_point=['5','6','8','9']
            candi_rdp=[]
            for j in range(len(feature_point_sets)):
                if (feature_point_sets[j].type in right_down_point) and feature_point_sets[j].y > feature_point_sets[i].y and ((feature_point_sets[j].x,feature_point_sets[j].y) in feature_point_sets[i].hori_intersect_line):
                    # if :
                    #     temp_grid.youxiajiao=feature_point_sets[j]
                    candi_rdp.append(feature_point_sets[j])
            temp_grid.youxiajiao=find_nearest(feature_point_sets[i],candi_rdp)



            left_up_point=['1','2','4','5']
            candi_lup=[]

            for k in range(0,i):
                if ((feature_point_sets[k].type) in left_up_point) and ((feature_point_sets[k].x,feature_point_sets[k].y) in feature_point_sets[i].verti_intersect_line):
                    candi_lup.append(feature_point_sets[k])
            temp_grid.zuoshangjiao=find_nearest(feature_point_sets[i],candi_lup)








            if (not temp_grid.zuoxiajiao==None) and (not temp_grid.youxiajiao==None) and (not temp_grid.zuoshangjiao==None):
                temp_grid.width = get_distance_feapoint(temp_grid.zuoxiajiao, temp_grid.youxiajiao)
                temp_grid.height = get_distance_feapoint(temp_grid.zuoxiajiao, temp_grid.zuoshangjiao)
                temp_grid.caculate_center()
                temp_grid.img_array = table_area[temp_grid.zuoshangjiao.x:temp_grid.zuoxiajiao.x + 1,
                                      temp_grid.zuoxiajiao.y:temp_grid.youxiajiao.y + 1]
                # 保存每个单元格
                # cv2.imwrite('grid_gray//%s.png' % (str(i)), temp_grid.img_array)
                # print('gridimg//%s  saved!'%(str(i)))
                i += 1

                # plt.imshow(temp_grid.img_array,cmap='gray')
                # plt.show()


                temp_grid.text_contents = ocr_grid_recognition(temp_grid.img_array)

                grid_list.append(temp_grid)

                        # self.hori_intersect_line = None
                        # self.verti_intersect_line = None


        else:
            continue
    return grid_list




def search_horizon_line(img,seed,parent_conn):
    print("Process start , searching horizontal line! ")
    width,height=img.shape

    def is_line_start_point(img_mat, x, y, detected_points,marker_img):
        """
        :param img_mat:
        :param x:
        :param y:
        :param detected_points:
        :return:
        """
        if marker_img[x,y]==1:
            return False

        width, length = img_mat.shape

        if y + SINGLE_GRID_WIDTH < width:
            if img_mat[x, y] == 255 and img_mat[x, y + 1] == 255 and img_mat[x, y + 2] == 255 and img_mat[
                x, y + 3] == 255 and img_mat[x, y + 4] == 255:
                if (y-5>=0 and img_mat[x,y-1]==0 and img_mat[x,y-2]==0 and img_mat[x,y-3]==0 and img_mat[x,y-4]==0 and img_mat[x,y-5]==0) or y<5:
                    for p in detected_points:
                        if get_distance((x, y), p) < 6:
                            return False
                    return True


                # if x-1>=0 and y-1>=0 and x+1<width and y+1<length:
                #     if not ((img_mat[x+1,y]==255 and img_mat[x+1,y+1]==255 and img_mat[x+1,y-1]==255) or (img_mat[x-1,y]==255 and img_mat[x-1,y+1]==255 and img_mat[x-1,y-1]==255)):
                #         for p in detected_points:
                #             if get_distance((x, y), p) < 4:
                #                 return False
            else:
                return False
        else:
            return False
        return False

    def search_next(img,poi,marker_img):
        assert marker_img.shape==img.shape,"标签矩阵和图像size不相同"

        x_length,y_length=img.shape

        if poi[1]+1 < y_length and img[poi[0],poi[1]+1]==255 :#and marker_img[poi[0],poi[1]+1]==0:
            return (poi[0],poi[1]+1)

        if poi[1]+2 < y_length and img[poi[0],poi[1]+2]==255 :#and marker_img[poi[0],poi[1]+2]==0:
            return (poi[0],poi[1]+1)

        if poi[1]+3 < y_length and img[poi[0],poi[1]+3]==255 :#and marker_img[poi[0],poi[1]+3]==0:
            return (poi[0],poi[1]+1)

        if poi[1]+4 < y_length and img[poi[0],poi[1]+4]==255 :#and marker_img[poi[0],poi[1]+4]==0:
            return (poi[0],poi[1]+1)


        if poi[1]+1 < y_length and poi[0]+1<x_length and img[poi[0]+1,poi[1]+1]==255 :#and marker_img[poi[0]+1,poi[1]+1]==0:
            return (poi[0]+1,poi[1]+1)

        if poi[1]+1 < y_length and poi[0]-1>=0 and img[poi[0]-1,poi[1]+1]==255 :#and marker_img[poi[0]-1,poi[1]+1]==0:
            return (poi[0]-1,poi[1]+1)

        if poi[1]+1 < y_length and poi[0]+2<x_length and img[poi[0]+2,poi[1]+1]==255:# and marker_img[poi[0]+2,poi[1]+1]==0:
            return (poi[0]+1,poi[1]+1)

        if poi[1]+1 < y_length and poi[0]-2>=0 and img[poi[0]-2,poi[1]+1]==255:# and marker_img[poi[0]-2,poi[1]+1]==0:
            return (poi[0]-1,poi[1]+1)

        if poi[1] + 2 < y_length and poi[0] + 1 < x_length and img[poi[0] + 1, poi[1] + 2] == 255 :#and marker_img[poi[0] + 1, poi[1] + 2]==0:
            return (poi[0], poi[1] + 1)
        if poi[1] + 2 < y_length and poi[0] - 1 >= 0 and img[poi[0] - 1, poi[1] + 2] == 255 :#and marker_img[poi[0] - 1, poi[1] + 2]==0:
            return (poi[0], poi[1] + 1)

        if poi[1]+2 <y_length and poi[0]+2 < x_length and img[poi[0]+2,poi[1]+2] ==255 :#and marker_img[poi[0]+2,poi[1]+2]==0:
            return (poi[0]+1,poi[1]+1)

        if poi[1]+2 <y_length and poi[0]-2 >=0 and img[poi[0]-2,poi[1]+2]==255 :#and marker_img[poi[0]-2,poi[1]+2]==0:
            return (poi[0]-1,poi[1]+1)


        if poi[1] + 3 < y_length and poi[0] + 1 < x_length and img[poi[0] + 1, poi[1] + 3] == 255 :#and marker_img[poi[0] + 1, poi[1] + 3]==0:
            return (poi[0], poi[1] + 1)
        if poi[1] + 3 < y_length and poi[0] - 1 >= 0 and img[poi[0] - 1, poi[1] + 3] == 255:# and marker_img[poi[0] - 1, poi[1] + 3]==0:
            return (poi[0], poi[1] + 1)

        if poi[1]+3 <y_length and poi[0]+2 < x_length and img[poi[0]+2,poi[1]+3] ==255 :#and marker_img[poi[0]+2,poi[1]+3]==0:
            return (poi[0],poi[1]+1)

        if poi[1]+3 <y_length and poi[0]-2 >=0 and img[poi[0]-2,poi[1]+3]==255:# and marker_img[poi[0]-2,poi[1]+3]==0:
            return (poi[0],poi[1]+1)

        if poi[1] + 4 < y_length and poi[0] + 1 < x_length and img[poi[0] + 1, poi[1] + 4] == 255 :#and marker_img[poi[0] + 1, poi[1] + 4]==0:
            return (poi[0], poi[1] + 1)
        if poi[1] + 4 < y_length and poi[0] - 1 >= 0 and img[poi[0] - 1, poi[1] + 4] == 255 :#and marker_img[poi[0] - 1, poi[1] + 4]==0:
            return (poi[0], poi[1] + 1)

        if poi[1] + 4 < y_length and poi[0] + 2 < x_length and img[poi[0] + 2, poi[1] + 4] == 255 :#and marker_img[poi[0] + 2, poi[1] + 4]==0:
            return (poi[0], poi[1] + 1)

        if poi[1] + 4 < y_length and poi[0] - 2 >= 0 and img[poi[0] - 2, poi[1] + 4] == 255 :#and marker_img[poi[0] - 2, poi[1] + 4]==0:
            return (poi[0], poi[1] + 1)
        return None

    def is_line_length_enough(line,thre=30):

        def line_var_x(line):
            x_lists=[k[0] for k in line]
            var=np.var(x_lists)
            return var
        p1=line[0]
        p2=line[len(line)-1]
        length=abs(p1[1]-p2[1])
        # length=np.sqrt(pow((p1[0]-p2[0]),2)+pow((p1[1]-p2[1]),2))
        if length > thre:   #and line_var_x(line) < 2.5:

            return True
        else:
            return False

    def is_line_merge(hori_lines, temp_line):
        #     ipdb.set_trace()
        for i in range(len(hori_lines)):
            # xcordi_sets=
            ite_line_hori_cord = np.mean([x[0] for x in hori_lines[i]])
            temp_line_hori_cord = np.mean([x[0] for x in temp_line])

            if abs(ite_line_hori_cord - temp_line_hori_cord) <= 30:  # 两条水平直线垂直距离小于15像素

                if temp_line[0][1] - hori_lines[i][-1][1] <= 20 and temp_line[0][1] - hori_lines[i][-1][1] >= 0:
                    # if abs(hori_lines[i][-1][1]-temp_line[0][1])<=10: #两条水平直线水平距离小于10 像素，两条直线连接起来
                    gap = abs(hori_lines[i][-1][1] - temp_line[0][1])
                    for j in range(gap - 1):
                        hori_lines[i].append((hori_lines[i][-1][0], hori_lines[i][-1][1] + 1))
                    hori_lines[i] += temp_line
                    return True
                elif hori_lines[i][0][1] - temp_line[-1][1] <= 20 and hori_lines[i][0][1] - temp_line[-1][1] > 0:

                    gap = abs(hori_lines[i][0][1] - temp_line[-1][1])
                    for j in range(gap - 1):
                        temp_line.append((temp_line[-1][0], temp_line[-1][1] + 1))
                    # showline([temp_line])
                    hori_lines[i] = temp_line + hori_lines[i]
                    return True

                elif hori_lines[i][0][1] <= temp_line[0][1] and hori_lines[i][-1][1] >= temp_line[0][1] and \
                        hori_lines[i][-1][1] <= temp_line[-1][1]:
                    for p in temp_line:
                        if p[1] > hori_lines[i][-1][1]:
                            hori_lines[i].append(p)
                    # hori_lines[i]=temp_line
                    return True

                elif temp_line[0][1] <= hori_lines[i][0][1] and temp_line[-1][1] >= hori_lines[i][0][1] and \
                        temp_line[-1][1] <= hori_lines[i][-1][1]:
                    for p in hori_lines[i]:
                        if p[1] > temp_line[-1][1]:
                            temp_line.append(p)
                    hori_lines[i] = temp_line
                    return True

                elif temp_line[0][1]<=hori_lines[i][0][1] and temp_line[-1][1]>=hori_lines[i][-1][1]:
                    hori_lines[i]=temp_line
                    return True
                elif hori_lines[i][0][1]<=temp_line[0][1] and hori_lines[i][-1][1]>=temp_line[-1][1]:
                    return True
            else:
                continue
        return False



    start_point=[]
    lines = []
    marker_img=np.zeros(shape=(width,height))

    for i in range(width):
        j=0
        while j <height-1:
            if is_line_start_point(img,i,j,start_point,marker_img):
                start_point.append((i,j))

                cache = search_next(img, (i, j),marker_img)
                temp_line = [(i,j)]
              #  marker_img[cache[0], cache[1]] = 1
                while not cache==None:
                    temp_line.append(cache)
                  #  marker_img[cache[0],cache[1]]=1
                    cache = search_next(img, cache,marker_img)
                    j+=1
                if cache==None:
                    j+=1
                # print('find %s points in temp line \n'%(str(len(temp_line_point))))
                if is_line_length_enough(temp_line):
                    if is_line_merge(lines,temp_line):
                        # print("Temp horizontal line Merged!")
                        pass
                    else:
                        lines.append(temp_line)
            else:
                j+=1

    # 可视化检测到的水平直线
    show_h_line=np.zeros(shape=(img.shape[0]+11,img.shape[1]+11),dtype=np.uint8)

    for m in lines:
        c_x=int(np.mean([i[0] for i in m]))
        c_y=int(np.mean([i[1] for i in m]))
        m.append((c_x+1,c_y))
        m.append((c_x + 2, c_y))
        m.append((c_x + 3, c_y))
        m.append((c_x + 4, c_y))
        m.append((c_x + 5, c_y))
        m.append((c_x + 6, c_y))
        m.append((c_x + 7, c_y))
        m.append((c_x + 8, c_y))
        m.append((c_x + 9, c_y))
        m.append((c_x + 10, c_y))

        for n in m:
            show_h_line[n[0],n[1]]=255
    # plt.imshow(show_h_line,cmap='gray')
    # plt.show()
    im=Image.fromarray(show_h_line)
    # im.show()
    im.save('检测到的水平线_标记直线.png')
    # return

    parent_conn.send(lines)
    parent_conn.close()



def search_vertical_line(img,seed,conn):
    print("process started ,, searching vertical line!")
    width, height = img.shape

    flag_line_start = False
    flag_line_end = False
    lines = []
    def is_line_start_point(img_mat, x, y, detected_points):
        """
        :param img_mat:
        :param x:
        :param y:
        :param detected_points:
        :return:
        """
        width, length = img_mat.shape

        if x + SINGLE_GRID_HEIGHT < width:
            if img_mat[x, y] == 255 and img_mat[x+1, y ] == 255 and img_mat[x+2, y] == 255 and img_mat[
                x+3, y ] == 255 and img_mat[x+4, y] == 255:
                if (x-5>=0 and img_mat[x-1,y]==0 and img_mat[x-2,y]==0 and img_mat[x-3,y]==0 and img_mat[x-4,y]==0 and img_mat[x-5,y]==0) or x<5:
                #
                # if x-1>=0 and y-1>=0 and x+1<width and y+1<length:
                #     if not ((img_mat[x,y+1]==255 and img_mat[x+1,y+1]==255 and img_mat[x-1,y+1]==255) or (img_mat[x,y-1]==255 and img_mat[x-1,y-1]==255 and img_mat[x+1,y-1]==255)):
                    for p in detected_points:
                        if get_distance((x, y), p) < 6:
                            return False
                    return True
            else:
                return False
        else:
            return False
        return False

    def search_next(img, poi):
        x_length, y_length = img.shape
        if poi[0] + 1 < x_length and img[poi[0]+1, poi[1]] == 255:
            return (poi[0]+1, poi[1] )
        if poi[0]+2<x_length and img[poi[0]+2,poi[1]]==255:
            return (poi[0]+1,poi[1])
        if poi[0]+3<x_length and img[poi[0]+3,poi[1]]==255:
            return (poi[0]+1,poi[1])
        if poi[0]+4<x_length and img[poi[0]+4,poi[1]]==255:
            return (poi[0]+1,poi[1])


        if poi[0] + 1 < x_length and poi[1] + 1 < y_length and img[poi[0] + 1, poi[1] + 1] == 255:
            return (poi[0] + 1, poi[1] + 1)
        if poi[0] + 1 < x_length and poi[1] - 1 >= 0 and img[poi[0] + 1, poi[1] - 1] == 255:
            return (poi[0] + 1, poi[1] - 1)
        if poi[0] + 1 < x_length and poi[1] + 2 < y_length and img[poi[0] + 1, poi[1] + 2] == 255:
            return (poi[0] + 1, poi[1] + 1)
        if poi[0] + 1 < x_length and poi[1] - 2 >= 0 and img[poi[0] + 1, poi[1] - 2] == 255:
            return (poi[0] + 1, poi[1] - 1)

        if poi[0]+2<x_length and poi[1]+1 <y_length and img[poi[0]+2,poi[1]+1]==255:
            return (poi[0]+1,poi[1])
        if poi[0]+2<x_length and poi[1]-1 >=0 and img[poi[0]+2,poi[1]-1]==255:
            return (poi[0]+1,poi[1])

        if poi[0]+2<x_length and poi[1]+2 <y_length and img[poi[0]+2,poi[1]+2]==255:
            return (poi[0]+1,poi[1]+1)

        if poi[0]+2<x_length and poi[0]-2 >=0 and img[poi[0]+2,poi[1]-2]==255:
            return (poi[0]+1,poi[1]-1)

        if poi[0] + 3 < x_length and poi[1] + 1 < y_length and img[poi[0] + 3, poi[1] + 1] == 255:
            return (poi[0] + 1, poi[1])
        if poi[0] + 3 < x_length and poi[1] - 1 >= 0 and img[poi[0] + 3, poi[1] - 1] == 255:
            return (poi[0] + 1, poi[1])

        if poi[0] + 3 < x_length and poi[1] + 2 < y_length and img[poi[0] + 3, poi[1] + 2] == 255:
            return (poi[0] + 1, poi[1] + 1)

        if poi[0] + 3 < x_length and poi[0] - 2 >= 0 and img[poi[0] + 3, poi[1] - 2] == 255:
            return (poi[0] + 1, poi[1] - 1)


        if poi[0] + 4 < x_length and poi[1] + 1 < y_length and img[poi[0] + 4, poi[1] + 1] == 255:
            return (poi[0] + 1, poi[1])
        if poi[0] + 4 < x_length and poi[1] - 1 >= 0 and img[poi[0] + 4, poi[1] - 1] == 255:
            return (poi[0] + 1, poi[1])

        if poi[0] + 4 < x_length and poi[1] + 2 < y_length and img[poi[0] + 4, poi[1] + 2] == 255:
            return (poi[0] + 1, poi[1] + 1)

        if poi[0] + 4 < x_length and poi[0] - 2 >= 0 and img[poi[0] + 4, poi[1] - 2] == 255:
            return (poi[0] + 1, poi[1] - 1)
        # if poi[0] +3 <x_length and img[poi[0]+3,poi[1]]==255:
        #     return (poi[0]+1,poi[1])
        #
        # if poi[0] +4 <x_length and img[poi[0]+4,poi[1]]==255:
        #     return (poi[0]+1,poi[1])


        return None

    def is_line_length_enough(line, thre=30):
        p1 = line[0]
        p2 = line[len(line) - 1]
        length = np.sqrt(pow((p1[0] - p2[0]), 2) + pow((p1[1] - p2[1]), 2))
        if length > thre:
            return True
        else:
            return False


    def is_line_merge(verti_lines, temp_line):
        #     ipdb.set_trace()
        for i in range(len(verti_lines)):
            # xcordi_sets=
            ite_line_hori_cord = np.mean([x[1] for x in verti_lines[i]])
            temp_line_hori_cord = np.mean([x[1] for x in temp_line])

            if abs(ite_line_hori_cord - temp_line_hori_cord) <= 40:  # 两条数值直线水平距离小于15像素

                if temp_line[0][0] - verti_lines[i][-1][0] <= 20 and temp_line[0][0] - verti_lines[i][-1][0] >= 0:
                    # if abs(hori_lines[i][-1][1]-temp_line[0][1])<=10: #两条水平直线水平距离小于10 像素，两条直线连接起来
                    gap = abs(verti_lines[i][-1][0] - temp_line[0][0])
                    for j in range(gap - 1):
                        verti_lines[i].append((verti_lines[i][-1][0]+1,  verti_lines[i][-1][1]))
                    verti_lines[i] += temp_line
                    return True
                elif verti_lines[i][0][0] - temp_line[-1][0] <= 20 and verti_lines[i][0][0] - temp_line[-1][0] > 0:
                    # 两条竖直直线水平距离小于 10 像素，连接两条直线

                    gap = abs(verti_lines[i][0][0] - temp_line[-1][0])
                    for j in range(gap - 1):
                        temp_line.append((temp_line[-1][0]+1, temp_line[-1][1]))
                    # showline([temp_line])
                    verti_lines[i] = temp_line + verti_lines[i]

                    return True

                elif verti_lines[i][0][0] <= temp_line[0][0] and verti_lines[i][-1][0] >= temp_line[0][0] and \
                        verti_lines[i][-1][0] <= temp_line[-1][0]:# 两条直线存在重叠区域取并集
                    for p in temp_line:
                        if p[0] > verti_lines[i][-1][0]:
                            verti_lines[i].append(p)
                    # hori_lines[i]=temp_line
                    return True

                elif temp_line[0][0] <= verti_lines[i][0][0] and temp_line[-1][0] >= verti_lines[i][0][0] and \
                        temp_line[-1][0] <= verti_lines[i][-1][0]:      #两条直线存在重叠区域，取并集

                    for p in verti_lines[i]:
                        if p[0] > temp_line[-1][0]:
                            temp_line.append(p)
                    verti_lines[i] = temp_line
                    return True
                elif temp_line[0][0]<=verti_lines[i][0][0] and temp_line[-1][0]>=verti_lines[i][-1][0]:
                    verti_lines[i]=temp_line
                    return True
                elif verti_lines[i][0][0]<=temp_line[0][0] and verti_lines[i][-1][0]>=temp_line[-1][0]:
                    return True
            else:
                continue

        # hori_lines.append(temp_line)
        return False




    start_point=[]
    for i in range(height):
        # print('searching the %s th row of the image\n=======================================' % (str(i)))
        j = 0
        while j < width - 1:
            if is_line_start_point(img,j,i,start_point):
                start_point.append((j,i))
                cache = search_next(img, (j, i))
                temp_line_point = [(j, i)]
                while not cache == None:
                    temp_line_point.append(cache)
                    cache = search_next(img, cache)
                    j += 1
                if cache == None:
                    j += 1
                # print('find %s points in temp line \n' % (str(len(temp_line_point))))

                if is_line_length_enough(temp_line_point):
                    if is_line_merge(lines,temp_line_point):
                        pass
                        # print("Temp vertical line Merged!")
                    else:
                        lines.append(temp_line_point)
                # if is_line_length_enough(temp_line_point):
                #     lines.append(temp_line_point)
            else:
                j += 1
    # v_lines+= lines
    #可视化检测到的竖直直线
    show_h_line = np.zeros(shape=img.shape, dtype=np.uint8)
    for m in lines:
        for n in m:
            show_h_line[n[0], n[1]] = 255
        # plt.imshow(show_h_line,cmap='gray')
        # plt.show()
    im = Image.fromarray(show_h_line)
    im.show()
    im.save('检测到的竖直线22.png')

    conn.send(lines)
    conn.close()
