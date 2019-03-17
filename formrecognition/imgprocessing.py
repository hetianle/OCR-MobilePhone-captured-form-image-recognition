
import json
from formrecognition.formvision import *
from datetime import datetime
from multiprocessing import Process,Queue,Pipe

from matplotlib import pyplot as plt

class Imgprocessing:

    def __init__(self,imgfile):

        if not imgfile:
            raise ValueError("parameter 'imgfile' must be a non-empty, non-None string")

        self.img_path=imgfile
        self.img_rgb=cv2.imread(self.img_path)
        self.img_rgb=self.resize(self.img_rgb)

        (self.img_width,self.img_length,self.img_depth)=self.img_rgb.shape

        self.grid_width=None
        self.grid_height=None

        print("Turn RGB image to gray!")
        self.img_gray=self.gray_scale(self.img_rgb)
        # cv2.imwrite('grayscale.png', self.img_gray)


        self.img_table_gray=None

        print('Image edge enhancement !')
        self.img_margin_dense=self.margin_dense(self.img_gray)

        # cv2.imwrite('生物量margin_dense.png', self.img_margin_dense)

        print("Turn gray image to binary image!")
        self.img_binary = self.binarization(self.img_margin_dense)

        # cv2.imwrite('bin.png', self.img_binary)

        print("Searching table areas!")
        self.img_table,self.img_table_gray = self.find_table_area(self.img_binary)

        # cv2.imwrite('_table_area_gray.png', self.img_table_gray)

        print("starting searching lines!!")


        self.lines=self.find_line(self.img_table)

      #  self.determain_grid_size(self.lines)
        print('Start searching cross points!')
        self.feature_point=self.search_feature_point(self.lines)

        print('Start searching grids!')
        self.grid_list=search_grid(self.feature_point,self.img_table_gray)

        self.save_to_json(self.grid_list)
        print('complete!!')


    def save_to_json(self,grids):
        verti_cordis,hori_cordis=[],[]
        for g in grids:
            flag=True
            flag_h=True

            for v in verti_cordis:
                if abs(v-g.zuoxiajiao.x)<6:
                    flag=False
                    break
            if flag:
                verti_cordis.append(g.zuoxiajiao.x)

            for h in hori_cordis:
                if abs(h-g.zuoxiajiao.y)<6:
                    flag_h=False
                    break
            if flag_h:
                hori_cordis.append((g.zuoxiajiao.y))

        verti_cordis.sort()
        hori_cordis.sort()

        def index(s, x):
            if x < s[0]:
                return -1
            for i in range(len(s)):
                if i + 1 < len(s):
                    if x >= s[i] and x < s[i + 1]:
                        return i
                elif i + 1 == len(s) and x >= s[-1]:
                    return i

        rows,columns=len(verti_cordis),len(hori_cordis)



        d={}
        d["rows"]=len(verti_cordis)
        d["columns"]=len(hori_cordis)
        d["contents"]=[]
        for g in grids:
            row_x=index(verti_cordis,g.center.x)
            col_y=index(hori_cordis,g.center.y)
            tg={}
            tg["row_num"]=row_x
            tg["col_num"]=col_y
            tg["text"]=g.text_contents
            d["contents"].append(tg)
        f = open('data.json', 'w')
        json.dump(d,f)
        f.close()
        print("Recognized data writed to ,'data.json',\n")


    def resize(self,img,length=2000):
        w, h, d = img.shape

        if w > h:
            new_w = length
            new_h = length / w * h
            print(w, h)
            print(new_w, new_h)

            print()
        else:
            new_h = length
            new_w = length / h * w
            print(w, h)
            print(new_w, new_h)

        newimg = cv2.resize(img, (int(new_h), int(new_w)), interpolation=cv2.INTER_CUBIC)
        # cv2.imwrite('/Users/tianle/Desktop/proj_p/生物量/生物量resize.png', newimg)
        # plt.imshow(newimg)
        # plt.show()
        return newimg


    def search_feature_point(self,lines):
        """
        dan执行时间:0:02:30.979977
        多线程执行时间:0:02:34.129947
        多线程执行时间:0:00:26.996367
        :param lines:
        :return:
        """
        h_lines=lines['horizontal']
        v_lines=lines['vertical']

        def cross_point(line1,line2):
            temp_fea_poi=None
            for p1 in line1:
                for p2 in line2:
                    if p1==p2:

                        temp_fea_poi=Feature_point(p1[0],p1[1])
                        temp_fea_poi.hori_intersect_line=line1
                        temp_fea_poi.verti_intersect_line=line2
                        determain_feature_point_type(temp_fea_poi)
                        return temp_fea_poi
            return None
                        # cross_points.append(temp_fea_poi)



        def process(h_lines,v_lines,conn):
            width,length=self.img_table.shape
            cross_point_list=[]
            print('Thread1 start')
            for h in h_lines:
                for v in v_lines:
                    temp_cross_poi=cross_point(h,v)
                    if temp_cross_poi==None:
                        if h[0][1]-5>=0:
                            h_left_r=5
                        else:
                            h_left_r=h[0][1]

                        if h[-1][1]+5<length:
                            h_right_r=5
                        else:
                            h_right_r=length-h[-1][1]-1

                        if v[0][0]-5>=0:
                            v_up_r=5
                        else:
                            v_up_r=v[0][0]

                        if v[-1][0]+5<width:
                            v_bottom_r=5
                        else:
                            v_bottom_r=width-v[-1][0]-1

                        hhh=[( h[0][0], h[0][1]- h_left_r+k) for k in range(h_left_r)]+ h + [(h[-1][0],h[-1][1]+k+1) for k in range(h_right_r)]
                        # print("左边增加像素：",h_left_r,"-----右边增加像素",h_right_r)

                        vvv=[(v[0][0]-v_up_r+k,v[0][1]) for k in range(v_up_r)]+ v +[(v[-1][0]+k+1,v[-1][1]) for k in range(v_bottom_r)]
                        # print("上边增加像素：", v_up_r, "-----下边增加像素", v_bottom_r)
                        temp_cross_poi=cross_point(hhh,vvv)
                        if not temp_cross_poi==None:
                            cross_point_list.append(temp_cross_poi)
                    else:
                        cross_point_list.append(temp_cross_poi)
                    # cross_point_list += cross_point(h, v)
                    # q.put(cross_point(h,v))
            conn.send(cross_point_list)
            conn.close()

        s = datetime.now()
       # poi1,poi2,poi3,poi4=Queue(),Queue(),Queue(),Queue()
        parent_conn1, child_conn1 = Pipe()
        parent_conn2, child_conn2 = Pipe()
        parent_conn3, child_conn3 = Pipe()
        parent_conn4, child_conn4 = Pipe()

        aquater=int(len(h_lines)/4)
        h_lines1=h_lines[0:aquater]
        h_lines2=h_lines[aquater:2*aquater]
        h_lines3=h_lines[2*aquater:3*aquater]
        h_lines4=h_lines[3*aquater:len(h_lines)]

        t1=Process(target=process,args=(h_lines1,v_lines,parent_conn1,))
        t2=Process(target=process,args=(h_lines2,v_lines,parent_conn2,))
        t3=Process(target=process,args=(h_lines3,v_lines,parent_conn3,))
        t4=Process(target=process,args=(h_lines4,v_lines,parent_conn4,))

        t1.start()
        t2.start()
        t3.start()
        t4.start()

        poi1=child_conn1.recv()
        poi2=child_conn2.recv()
        poi3=child_conn3.recv()
        poi4=child_conn4.recv()
        t1.join()
        t2.join()
        t3.join()
        t4.join()

        #
        # while True:
        #     if t1.is_alive()==False and t2.is_alive()==False and t3.is_alive()==False and t4.is_alive()==False:
        #         e = datetime.now()
        #         print('多线程执行时间:%s' % (str(e - s)))
        #         break



        cross_point_list=poi1+poi2+poi3+poi4
        e = datetime.now()
        print('多线程执行时间:%s' % (str(e - s)))

        print(len(cross_point_list))

        simplified_cross_point=[]

        def is_repeated(p,points):
            for k in points:
                a=(p.x,p.y)
                b=(k.x,k.y)
                dis=get_distance(a,b)
                if dis < 10:
                    return True
                else:
                    continue
            return False

        # #
        #img_cross_point = np.zeros(shape=self.img_table.shape, dtype=np.uint8)
        #
        # #
        for i in cross_point_list:
            if not is_repeated(i,simplified_cross_point):
                simplified_cross_point.append(i)


        ## 保存特征点图片

        #         if i.type=='1':
        #             img_cross_point[i.x+1,i.y]=255
        #             img_cross_point[i.x + 2, i.y] = 255
        #             img_cross_point[i.x + 3, i.y] = 255
        #
        #             img_cross_point[i.x, i.y+1] = 255
        #             img_cross_point[i.x, i.y+2] = 255
        #             img_cross_point[i.x, i.y+3] = 255
        #         elif i.type=='2':
        #             img_cross_point[i.x + 1, i.y] = 255
        #             img_cross_point[i.x + 2, i.y] = 255
        #             img_cross_point[i.x + 3, i.y] = 255
        #
        #             img_cross_point[i.x, i.y + 1] = 255
        #             img_cross_point[i.x, i.y + 2] = 255
        #             img_cross_point[i.x, i.y + 3] = 255
        #
        #
        #             img_cross_point[i.x, i.y - 1] = 255
        #             img_cross_point[i.x, i.y - 2] = 255
        #             img_cross_point[i.x, i.y - 3] = 255
        #         elif i.type=='3':
        #             img_cross_point[i.x + 1, i.y] = 255
        #             img_cross_point[i.x + 2, i.y] = 255
        #             img_cross_point[i.x + 3, i.y] = 255
        #
        #
        #
        #             img_cross_point[i.x, i.y - 1] = 255
        #             img_cross_point[i.x, i.y - 2] = 255
        #             img_cross_point[i.x, i.y - 3] = 255
        #         elif i.type=='4':
        #
        #             img_cross_point[i.x + 1, i.y] = 255
        #             img_cross_point[i.x + 2, i.y] = 255
        #             img_cross_point[i.x + 3, i.y] = 255
        #
        #             img_cross_point[i.x - 1, i.y] = 255
        #             img_cross_point[i.x - 2, i.y] = 255
        #             img_cross_point[i.x - 3, i.y] = 255
        #             img_cross_point[i.x, i.y + 1] = 255
        #             img_cross_point[i.x, i.y + 2] = 255
        #             img_cross_point[i.x, i.y + 3] = 255
        #
        #         elif i.type=='5':
        #             img_cross_point[i.x + 1, i.y] = 255
        #             img_cross_point[i.x + 2, i.y] = 255
        #             img_cross_point[i.x + 3, i.y] = 255
        #
        #             img_cross_point[i.x - 1, i.y] = 255
        #             img_cross_point[i.x - 2, i.y] = 255
        #             img_cross_point[i.x - 3, i.y] = 255
        #
        #
        #             img_cross_point[i.x, i.y + 1] = 255
        #             img_cross_point[i.x, i.y + 2] = 255
        #             img_cross_point[i.x, i.y + 3] = 255
        #
        #             img_cross_point[i.x, i.y - 1] = 255
        #             img_cross_point[i.x, i.y - 2] = 255
        #             img_cross_point[i.x, i.y - 3] = 255
        #         elif i.type=='6':
        #             img_cross_point[i.x + 1, i.y] = 255
        #             img_cross_point[i.x + 2, i.y] = 255
        #             img_cross_point[i.x + 3, i.y] = 255
        #
        #             img_cross_point[i.x - 1, i.y] = 255
        #             img_cross_point[i.x - 2, i.y] = 255
        #             img_cross_point[i.x - 3, i.y] = 255
        #
        #             img_cross_point[i.x, i.y - 1] = 255
        #             img_cross_point[i.x, i.y - 2] = 255
        #             img_cross_point[i.x, i.y - 3] = 255
        #         elif i.type=='7':
        #             img_cross_point[i.x - 1, i.y] = 255
        #             img_cross_point[i.x - 2, i.y] = 255
        #             img_cross_point[i.x - 3, i.y] = 255
        #
        #             img_cross_point[i.x, i.y + 1] = 255
        #             img_cross_point[i.x, i.y + 2] = 255
        #             img_cross_point[i.x, i.y + 3] = 255
        #         elif i.type=='8':
        #
        #             img_cross_point[i.x - 1, i.y] = 255
        #             img_cross_point[i.x - 2, i.y] = 255
        #             img_cross_point[i.x - 3, i.y] = 255
        #             img_cross_point[i.x, i.y + 1] = 255
        #             img_cross_point[i.x, i.y + 2] = 255
        #             img_cross_point[i.x, i.y + 3] = 255
        #
        #             img_cross_point[i.x, i.y - 1] = 255
        #             img_cross_point[i.x, i.y - 2] = 255
        #             img_cross_point[i.x, i.y - 3] = 255
        #         elif i.type=='9':
        #
        #             img_cross_point[i.x - 1, i.y] = 255
        #             img_cross_point[i.x - 2, i.y] = 255
        #             img_cross_point[i.x - 3, i.y] = 255
        #
        #             img_cross_point[i.x, i.y - 1] = 255
        #             img_cross_point[i.x, i.y - 2] = 255
        #             img_cross_point[i.x, i.y - 3] = 255
        #
        # im=Image.fromarray(img_cross_point)
        # im.save('not_sim_featurepoint4.png')


        # print("Simplefied cross points number: %s \n 生物量中间图像/not_sim_featurepoint.png saved!"%(str(len(simplified_cross_point))))
        return simplified_cross_point


       # return cross_point_list



    def find_line(self,img):

        parent_conn1, child_conn1 = Pipe()
        parent_conn2, child_conn2 = Pipe()

        # # search_vertical_line(img,0,parent_conn2)
        # search_horizon_line(img, 0, parent_conn1)
        # return

        search_hline_process=Process(target=search_horizon_line,args=(img,0,parent_conn1,))
        search_vline_process=Process(target=search_vertical_line,args=(img,0,parent_conn2,))

        search_hline_process.start()
        search_vline_process.start()

        h_lines =child_conn1.recv()
        v_lines =child_conn2.recv()
        search_hline_process.join()
        search_vline_process.join()
        # print(search_hline_process.isAlive,search_vline_process.isAlive)

        assert (not search_hline_process.is_alive()) and (not search_vline_process.is_alive()),"直线搜索线程出现异常,异步线程没有结束"

        # im=np.zeros(shape=self.img_table.shape,dtype=np.uint8)
        # for i in h_lines:
        #     for j in i:
        #         im[j[0],j[1]]=255
        # for i in v_lines:
        #     for j in i:
        #         im[j[0], j[1]] = 255
        #
        # imggg=Image.fromarray(im)
        # imggg.save('/Users/tianle/Desktop/proj_p/生物量中间图像/竖直和水平直线.png')

        return {'horizontal': h_lines, 'vertical': v_lines}


    def find_table_area(self,img):
        c=0
        assert img.ndim==2,"定位图像中表格位置时发现错误，预处理后的图像不是二维图像"
        img_table=None
        width,length=img.shape

        stack=[]
        lab_matrix=np.zeros(shape=img.shape,dtype=np.int)

        def recursive_search(img,lab_matrix,stack,temp_label,c):
            c+=1
            print('recursive %sth /\n'%(str(c)),str(len(stack)),'\n')
            if len(stack)==0:
                return None
            i,j=stack[len(stack)-1]
            del stack[len(stack)-1]
          #  assert img[i,j]==255,""
           # assert lab_matrix[i,j]==0,""
            lab_matrix[i,j]=temp_label

            if i - 1 >= 0 and img[i - 1, j] == 255 and lab_matrix[i - 1, j] == 0:
                stack.append((i - 1, j))

            if j + 1 < length and img[i, j + 1] == 255 and lab_matrix[i, j + 1] == 0:
                stack.append((i, j + 1))

            if i + 1 < width and img[i + 1, j] == 255 and lab_matrix[i + 1, j] == 0:
                stack.append((i + 1, j))

            if j - 1 >= 0 and img[i, j - 1] == 255 and lab_matrix[i, j - 1] == 0:
                stack.append((i, j - 1))
            return recursive_search(img,lab_matrix,stack,temp_label,c)


            #if img[i,j]==255 and lab_matrix[i,j]==0:

        temp_label = 2
        for i in range(int(width)):
            for j in range(int(length)):

                if img[i,j]==255 and lab_matrix[i,j]==0:
                    temp_seed=(i,j)
                    # lab_matrix[i,j]=temp_label
                    #
                    # if i-1>=0 and img[i-1,j]==255 and lab_matrix[i-1,j]==0:
                    #     stack.append((i-1,j))
                    #
                    # if j+1<length and img[i,j+1]==255 and lab_matrix[i,j+1]==0:
                    #     stack.append((i,j+1))
                    #
                    # if i+1<width and img[i+1,j]==255 and lab_matrix[i+1,j]==0:
                    #     stack.append((i+1,j))
                    #
                    # if j-1>=0 and img[i,j-1]==255 and temp_label[i,j-1]==0:
                    #     stack.append((i,j-1))

                    stack.append((i,j))

                    while not len(stack)==0:
                        m,n=stack.pop()
                        lab_matrix[m,n]=temp_label
                        if m - 1 >= 0 and img[m - 1, n] == 255 and lab_matrix[m - 1, n] == 0:
                            stack.append((m - 1,n))
                        elif m-2>=0 and img[m-2,n]==255 and lab_matrix[m-2,n]==0:
                            stack.append((m-1,n))
                        elif m-3>=0 and img[m-3,n]==255 and lab_matrix[m-3,n]==0:
                            stack.append((m-1,n))

                        elif m-4>=0 and img[m-4,n]==255 and lab_matrix[m-4,n]==0:
                            stack.append((m-1,n))


                        if n + 1 < length and img[m, n + 1] == 255 and lab_matrix[m, n + 1] == 0:
                            stack.append((m, n + 1))
                        elif n + 2 < length and img[m, n + 2] == 255 and lab_matrix[m, n + 2] == 0:
                            stack.append((m, n + 1))
                        elif n + 3 < length and img[m, n + 3] == 255 and lab_matrix[m, n + 3] == 0:
                            stack.append((m, n + 1))
                        elif n + 4 < length and img[m, n + 4] == 255 and lab_matrix[m, n + 4] == 0:
                            stack.append((m, n + 1))

                        if m + 1 < width and img[m + 1, n] == 255 and lab_matrix[m + 1, n] == 0:
                            stack.append((m + 1, n))
                        elif m + 2 < width and img[m + 2, n] == 255 and lab_matrix[m + 2, n] == 0:
                            stack.append((m + 1, n))
                        elif m + 3 < width and img[m + 3, n] == 255 and lab_matrix[m + 3, n] == 0:
                            stack.append((m + 1, n))
                        elif m + 4 < width and img[m + 4, n] == 255 and lab_matrix[m + 4, n] == 0:
                            stack.append((m + 1, n))

                        if n- 1 >= 0 and img[m, n - 1] == 255 and lab_matrix[m, n - 1] == 0:
                            stack.append((m, n - 1))
                        elif n- 2 >= 0 and img[m, n - 2] == 255 and lab_matrix[m, n - 2] == 0:
                            stack.append((m, n - 1))
                        elif n- 3 >= 0 and img[m, n - 3] == 255 and lab_matrix[m, n - 3] == 0:
                            stack.append((m, n - 1))
                        elif n- 4 >= 0 and img[m, n - 4] == 255 and lab_matrix[m, n - 4] == 0:
                            stack.append((m, n - 1))


                    # lab_matrix[i,j]=1
                    # if recursive_search(img,lab_matrix,stack,temp_label,c)==None:
                    #     temp_label+=1
                    assert len(stack)==0,"当前stack 不为0！！"
                   # print('find %sth connected area!\n'%(str(temp_label)))
                    temp_label+=1

        # plt.imshow(lab_matrix,cmap='gray')
        # plt.show()


        lab_mat_line=lab_matrix.ravel()
        # hist_value,bin_edge=np.histogram(lab_mat_line,bins=len(set(lab_mat_line.tolist())))
        # max_connecity_label=hist_value.tolist().index(hist_value.max())
      #  hist_value

        unique_label,count_label=np.unique(lab_mat_line,return_counts=True)
        assert len(unique_label)==len(count_label),"计算每个连通区域的大小时出现错误"

        unique_count=[]
        for ite in range(len(unique_label)):
            temp_label_positions=np.where(lab_matrix==unique_label[ite])
            assert count_label[ite]==len(temp_label_positions[0]) and len(temp_label_positions[0])==len(temp_label_positions[1]),"当前联通分量的标签的数量，和所有该标签像素的位置的数量必须相等"

            xmin, ymin, xmax, ymax = 100000, 100000, 0, 0
            for k in range(len(temp_label_positions[0])):
                if temp_label_positions[0][k] < xmin:
                    xmin = temp_label_positions[0][k]
                if temp_label_positions[1][k] < ymin:
                    ymin = temp_label_positions[1][k]
                if temp_label_positions[0][k] > xmax:
                    xmax = temp_label_positions[0][k]
                if temp_label_positions[1][k] > ymax:
                    ymax = temp_label_positions[1][k]
            # temp_connectity_dis=np.sqrt((xmin - xmax) * (xmin - xmax) + (ymin - ymax) * (ymin - ymax))
            unique_count.append((unique_label[ite],count_label[ite],temp_label_positions,(xmin,ymin,xmax,ymax)))

        def sortby_dig_dis(count):
            return count[1]

        sorted_unique_count=sorted(unique_count,key=sortby_dig_dis,reverse=True)

        # dict_count=dict(zip(*np.unique(lab_mat_line, return_counts=True)))
        # dict_count_sorted=[v for v in sorted(dict_count.values(),reverse=True)]

        # candidates=sorted_unique_count[0:10]
        connectity_form=None
        if sorted_unique_count[0][0]==0 and not sorted_unique_count[1][0]==0:
            connectity_form=sorted_unique_count[1]
        if not sorted_unique_count[0][0]==0 and sorted_unique_count[1][0]==0:
            connectity_form = sorted_unique_count[0]

        if connectity_form==None:
            raise ValueError("Connectity form not found!")

        # for candi in candidates:
        #     if not candi[0]==0 and :
        # connectity_form=np.zeros(shape=(width,length),dtype=np.uint8)
        # for u in range(width):
        #     for v in range(length):
        #         if lab_matrix[u,v] in candidates_label:
        #             connectity_form[u,v]=255


        #
        # plt.imshow(connectity_form,cmap='gray')
        # plt.show()
        assert connectity_form[3][2]+1 > connectity_form[3][0],''
        assert connectity_form[3][2]+1 <=width,""

        assert connectity_form[3][3]+1>connectity_form[3][1],''
        assert connectity_form[3][3]+1<=length,""



        img_table=img[connectity_form[3][0]:connectity_form[3][2]+1,connectity_form[3][1]:connectity_form[3][3]+1]
        assert img.shape==self.img_gray.shape,"图像预处理后shape应该保持不变"
        img_table_gray=self.img_gray[connectity_form[3][0]:connectity_form[3][2]+1,connectity_form[3][1]:connectity_form[3][3]+1]

        # plt.imshow(self.img_table_gray,cmap='gray')
        # plt.show()
        return img_table,img_table_gray


    def gray_scale(self,img):
       # img = self.img_rgb
        width, length, depth = img.shape
        red_img = img[:, :, 2]
        green_img = img[:, :, 1]
        blue_img = img[:, :, 0]
        gray = np.ndarray(shape=(width, length), dtype=np.uint8)
        for i in range(width):
            for j in range(length):
                gray[i][j] = 0.3 * red_img[i][j] + 0.59 * green_img[i][j] + 0.11 * blue_img[i][j]

        return gray


    def margin_dense(self,img,kernal=1):
        kernal=int(kernal)
        half_k=int(kernal/2)

        ###sobel kernel

        img_sobelx=cv2.Sobel(img,cv2.CV_8U,1,0,ksize=kernal)
        img_sobely=cv2.Sobel(img,cv2.CV_8U,0,1,ksize=kernal)
        img_sobel_general=img_sobely+img_sobelx
        img_edge_dense=np.ndarray(shape=(self.img_width,self.img_length),dtype=np.float32)

        for i in range(self.img_width):
            for j in range(self.img_length):
                if i-half_k>=0 and i+1+half_k<=self.img_width:
                    temp=img_sobel_general[i-half_k:i +1+half_k, j:j+1]
                    l=np.sum(temp)
                    img_edge_dense[i,j]=l
                    # print()

                elif i-half_k>=0 and i+1+half_k > self.img_width:
                    temp=img_sobel_general[i-half_k:self.img_width, j:j+1]
                    img_edge_dense[i,j]=kernal*np.mean(temp)


                elif i-half_k<0 and i+1+half_k <= self.img_width:
                    temp = img_sobel_general[0:i+1+half_k, j:j + 1]
                    img_edge_dense[i, j] = kernal * np.mean(temp)
                else:
                    assert i-half_k < 0 and i+1+half_k > self.img_width, "Something wrong happen when calcu margin dense"


        img_edge_dense=np.uint8(img_edge_dense)
        max=np.max(img_edge_dense)
        min=np.min(img_edge_dense)
        return img_edge_dense


    def binarization(self,img):
       # ret2, th2 = cv2.threshold(img_edge_dense, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

        # plt.imshow(th2,cmap='gray')
        # plt.show()
        return th2
