import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from project.class02_Crawling_Rich import CrawlingRich
from project.class03_Save_Image import SaveImage

warnings.filterwarnings(action='ignore')
from math import pi
import matplotlib.font_manager as fm

class DecagonPlot:
    # base_path = r"C:\labs\\project\\project\\celeba-dataset"
    def __init__(self, base_path):
        self.base_path = base_path

    def return_type(self, name_number, title=True):
        self.name_number = name_number
        self.title = title
        # [8, 4, 0, 7, 9, 3, 10, 15, 12, 11]
        si = SaveImage(self.base_path)
        new_df = si.test_image_df_recall(name_number)[
            ['eye_between', 'left_eye_width', 'right_eye_width', 'left_eyebrow', 'nose_width',
             'right_eyebrow', 'nose_height', 'jaw_height', 'mouth_width', 'philtrum']]
        new_df = new_df.values.tolist()[0]

        cr = CrawlingRich(self.base_path)
        data = cr.rich_total_dataframe()
        blist = []

        for x in range(496):
            slist = []
            for y in range(10):
                slist.append(data.iloc[x][y])
            blist.append(slist)

        clist = []
        for i in range(496):
            sum = 0
            for j in range(10):
                sum += abs(blist[i][j] - new_df[j])
            clist.append(sum)

        # return data.index[np.argmin(clist)]     # 부자이름
        # 0 자수성가형, 1 금수저형, 2 투자의귀재, 3 또라이형, 4 자퇴형
        # 5 결혼형, 6 시인형, 7 UN특사형, 8 정치인형, 9 professor type
        t = int(data.iloc[np.argmin(clist), 10])
        print(t)
        print(f'당신과 가장 닮은 부자는 {data.index[np.argmin(clist)]}입니다.')
        if title == True:
            if t == 0:
                return (f'자수성가형!\n뼈빠지게 고생하는 타입인 당신!\n언젠간 포브스 500에 당신의 이름이 실릴 날이...\n(ex. 제프 베조스)')
            elif t == 1:
                return (f'금수저형!\n꽝! 다음 생애에... (ex. 이건희)')
            elif t == 2:
                return (f'투자의귀재!\n보는 눈이 있는 당신!\n손대는 것마다 투자성공!\n(ex. 워렌 버핏)')
            elif t == 3:
                return (f'또라이형!\n 모 아니면 도! 포브스와 정신병원의 갈림길에 서 있는 당신!\n(ex. 일론머스크)')
            elif t == 4:
                return (f'자퇴형!\n 일단 자퇴를 해라. \n그러면 성공할 것이다.\n(ex. 주커버그, 빌게이츠, ...)')
            elif t == 5:
                return (f'결혼형!\n 배우자 복이 있는 당신!\n 행복하세요! (ex. 맥킨지 베조스)')
            elif t == 6:
                return (f'시인형!\n 당신은 부자의 관상은 아닙니다.. 하지만!\n17세기 최고의 시인이 될 수 있는 관상!\n돈보다 문학을 선택한 당신! 화이팅! (ex. 월터스콧)')
            elif t == 7:
                return (
                    f'UN특사형!\n 당신은 부자의 관상은 아닙니다.. 하지만!\n국제무대의 비둘기가 될 수 있는 관상!\nun특사에 도전해보는 것은 어떨까요?! (ex. Peter Thomson)')
            elif t == 8:
                return (
                    f'정치인형!\n 당신은 부자의 관상은 아닙니다.. 하지만!\n뒷주머니로 챙기기 나름!\n 정치로 세계 500대 부자에 도전해 보세요! \n(ex.Elaine Marshall)')
            elif t == 9:
                return (
                    f'professor type!\n 당신은 부자의 관상이 아닙니다..\n 하지만! 대학원 5년을 견디면...\n 교수가 될, 수도 있는 당신! 화이팅..\n (ex. Mary Malone)')
        elif title == False:
            return (f'당신과 가장 닮은 부자는 {data.index[np.argmin(clist)]}입니다.')

    def plot_radar_chart(self, name_number=1000, percent=False):
        self.name_number = name_number
        self.percent = percent
        rich_mean = [0.257579, 0.162942, 0.162557, 0.272610, 0.202978,
                     0.265984, 0.290559, 0.268134, 0.381116, 0.107953]

        if name_number == 1000:  # isinstance(name_number, list)
            new_df = rich_mean
        else:
            si = SaveImage(self.base_path)
            new_df = si.test_image_df_recall(name_number)[
                ['eye_between', 'left_eye_width', 'right_eye_width', 'left_eyebrow', 'nose_width',
                 'right_eyebrow', 'nose_height', 'jaw_height', 'mouth_width', 'philtrum']]
            new_df = new_df.values.tolist()[0]

        dif = []
        for x in range(10):
            dif.append(round((1 - abs(rich_mean[x] - new_df[x]) / rich_mean[x]) * 100, 2))

        # Font
        fm.get_fontconfig_fonts()
        font_location = 'C:/windows/fonts/gaesung.ttf'  # For Windows
        font_name = fm.FontProperties(fname=font_location).get_name()
        plt.rc('font', family=font_name, size=10)

        # cat = ['eye_between', 'left_eye_width', 'right_eye_width', 'left_eyebrow', 'nose_width',
        #        'right_eyebrow', 'nose_height', 'jaw_height', 'mouth_width', 'philtrum']
        cat = ['미간', '좌안 너비', '우안 너비', '왼눈썹 너비', '코 너비',
               '오른눈썹 너비', '코 길이', '턱 길이', '입 너비', '인중 길이']
        values = dif

        N = len(cat)

        x_as = [n / float(N) * 2 * pi for n in range(N)]

        # Because our chart will be circular we need to append a copy of the first
        # value of each list at the end of each list with data
        values += values[:1]
        x_as += x_as[:1]
        # print(x_as)

        # Set color of axes
        plt.rc('axes', linewidth=0, edgecolor="#888888")

        # Create polar plot
        fig, ax = plt.subplots(nrows=1, ncols=2)

        ax[0] = plt.subplot(121, polar=True)

        # Set clockwise rotation. That is:
        ax[0].set_theta_offset(pi / 2)
        ax[0].set_theta_direction(-1)

        # Set position of y-labels
        ax[0].set_rlabel_position(0)

        # Set color and linestyle of grid
        ax[0].xaxis.grid(True, color="#888888", linestyle='dashed', linewidth=0.5)
        ax[0].yaxis.grid(True, color="#888888", linestyle='dashed', linewidth=0.5)

        # Set number of radial axes and remove labels
        plt.xticks(x_as[:-1], [])

        # Set yticks
        plt.yticks([50, 60, 70, 80, 90], ["50", "60", "70", "80", "90"])

        # Plot data
        # 부자 사진
        ax[0].plot(x_as, [100 * x / x for x in range(1, 12)], color='#0099a4', linewidth=2, linestyle='solid', zorder=3)
        ax[0].fill(x_as, [100 * x / x for x in range(1, 12)], color='#0099a4', alpha=0.0)

        # 일반 사진
        ax[0].plot(x_as, values, color='#f29886', linewidth=1, linestyle='solid', zorder=3, marker='o', markersize=4)
        ax[0].fill(x_as, values, color='#f29886', alpha=0.3)

        plt.ylim(50, 100)

        if percent:
            ax[0].set_title(f'{round(np.mean(dif), 4)} %', size=18,
                            color='red', rotation='horizontal', loc='center')

            # Draw ytick labels to make sure they fit properly
        for i in range(N):
            angle_rad = i / float(N) * 2 * pi
            if angle_rad == 0:
                ha, distance_ax = "center", 103
            elif 0 < angle_rad < pi:
                ha, distance_ax = "left", 101
            elif angle_rad == pi:
                ha, distance_ax = "center", 105
            else:
                ha, distance_ax = "right", 101

            ax[0].text(angle_rad, distance_ax, cat[i], size=12, horizontalalignment=ha, verticalalignment="center",
                       fontstyle='oblique')

        if name_number == 1000:
            pass

        else:
            ax[1].axis('off')
            ax[1].axis('tight')
            # clust_data = ['eye_between', 'left_eye_width', 'right_eye_width', 'left_eyebrow', 'nose_width',
            #               'right_eyebrow', 'nose_height', 'jaw_height', 'mouth_width', 'philtrum']
            clust_data = ['미간', '좌안 너비', '우안 너비', '왼눈썹 너비', '코 너비',
                          '오른눈썹 너비', '코 길이', '턱 길이', '입 너비', '인중 길이']
            dif = dif[:-1]
            percent_data = dif

            # collabel_1 = ['location']
            collabel_1 = ['위치']
            df = pd.DataFrame(clust_data, columns=collabel_1)
            # df['percent(%)'] = percent_data
            df['퍼센트(%)'] = percent_data
            print(df)
            ax[1].table(cellText=df.values, colLabels=df.columns, loc='center',
                        cellColours=[[('#66cccc'), ('#ffcccc')],
                                     [('#66cccc'), ('#ffcccc')],
                                     [('#66cccc'), ('#ffcccc')],
                                     [('#66cccc'), ('#ffcccc')],
                                     [('#66cccc'), ('#ffcccc')],
                                     [('#66cccc'), ('#ffcccc')],
                                     [('#66cccc'), ('#ffcccc')],
                                     [('#66cccc'), ('#ffcccc')],
                                     [('#66cccc'), ('#ffcccc')],
                                     [('#66cccc'), ('#ffcccc')]],
                        colWidths=(0.5, 0.3),
                        colLoc='center', cellLoc='center', FONTSIZE=20)
            ax[1].set_title(f'{self.return_type(name_number)}', size=15, rotation='horizontal', loc='center')
            ax[1].text(x=-0.035, y=-0.035, s=self.return_type(name_number, title=False), size=15,
                       horizontalalignment='center', verticalalignment="center", fontstyle='oblique')



if __name__ == '__main__':
    dp = DecagonPlot(r"C:\labs\\project\\project\\celeba-dataset")
    dp.plot_radar_chart(name_number=1014, percent=True)
    # plt.tight_layout(h_pad=0.2, w_pad=0.2,
    #                  rect=(0.11, 0.11, 0.9, 0.88))
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()
    # cr = CrawlingRich(r"C:\labs\\project\\project\\celeba-dataset")
    # data = cr.rich_total_dataframe()
    # print(data)
    # print(data.shape)
