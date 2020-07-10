import pandas as pd
from math import pi
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

from project.rich_dataframe_final import rich_total_dataframe
from project.test_img_total_code_json_keras import test_image_df_recall



def return_type(name_number):
    # [8, 4, 0, 7, 9, 3, 10, 15, 12, 11]
    new_df = test_image_df_recall(name_number)[
        ['eye_between', 'left_eye_width', 'right_eye_width', 'left_eyebrow', 'nose_width',
         'right_eyebrow', 'nose_height', 'jaw_height', 'mouth_width', 'philtrum']]
    new_df = new_df.values.tolist()[0]

    data = rich_total_dataframe()
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
    # print(data.index[np.argmin(clist)])
    t = int(data.iloc[np.argmin(clist), 10])
    print(t)
    print(f'당신과 가장 닮은 부자는 {data.index[np.argmin(clist)]}입니다.')
    if t == 0:
        return(f'자수성가형!\n뼈빠지게 고생하는 타입인 당신!\n언젠간 포브스 500에 당신의 이름이 실릴 날이...\n(ex. 제프 베조스)\n당신과 가장 닮은 부자는 {data.index[np.argmin(clist)]}입니다.')
    elif t == 1:
        return(f'금수저형!\n꽝! 다음 생애에... (ex. 이건희)\n당신과 가장 닮은 부자는 {data.index[np.argmin(clist)]}입니다.')
    elif t == 2:
        return(f'투자의귀재!\n보는 눈이 있는 당신!\n손대는 것마다 투자성공!\n(ex. 워렌 버핏)\n당신과 가장 닮은 부자는 {data.index[np.argmin(clist)]}입니다.')
    elif t == 3:
        return(f'또라이형!\n 모 아니면 도! 포브스와 정신병원의 갈림길에 서 있는 당신!\n(ex. 일론머스크)\n당신과 가장 닮은 부자는 {data.index[np.argmin(clist)]}입니다.')
    elif t == 4:
        return(f'자퇴형!\n 일단 자퇴를 해라. \n그러면 성공할 것이다.\n(ex. 주커버그, 빌게이츠, ...)\n당신과 가장 닮은 부자는 {data.index[np.argmin(clist)]}입니다.')
    elif t == 5:
        return(f'결혼형!\n 배우자 복이 있는 당신!\n 행복하세요! (ex. 맥킨지 베조스)\n당신과 가장 닮은 부자는 {data.index[np.argmin(clist)]}입니다.')
    elif t == 6:
        return(f'시인형!\n 당신은 부자의 관상은 아닙니다.. 하지만!\n17세기 최고의 시인이 될 수 있는 관상!\n돈보다 문학을 선택한 당신! 화이팅! (ex. 월터스콧)\n당신과 가장 닮은 부자는 {data.index[np.argmin(clist)]}입니다.')
    elif t == 7:
        return(f'UN특사형!\n 당신은 부자의 관상은 아닙니다.. 하지만!\n국제무대의 비둘기가 될 수 있는 관상!\nun특사에 도전해보는 것은 어떨까요?! (ex. Peter Thomson)\n당신과 가장 닮은 부자는 {data.index[np.argmin(clist)]}입니다.')
    elif t == 8:
        return(f'정치인형!\n 당신은 부자의 관상은 아닙니다.. 하지만!\n뒷주머니로 챙기기 나름!\n 정치로 세계 500대 부자에 도전해 보세요! \n(ex.Elaine Marshall)\n당신과 가장 닮은 부자는 {data.index[np.argmin(clist)]}입니다.')
    elif t == 9:
        return(f'professor type!\n 당신은 부자의 관상이 아닙니다..\n 하지만! 대학원 5년을 견디면...\n 교수가 될, 수도 있는 당신! 화이팅..\n (ex. Mary Malone)\n당신과 가장 닮은 부자는 {data.index[np.argmin(clist)]}입니다.')


def plot_radar_chart(name_number=1000,
                     color='r', percent=False):
    rich_mean = [0.257579, 0.162942, 0.162557, 0.272610, 0.202978,
                 0.265984, 0.290559, 0.268134, 0.381116, 0.107953]

    if name_number == 1000:  # isinstance(name_number, list)
        new_df = rich_mean
    else:
        new_df = test_image_df_recall(name_number)[
            ['eye_between', 'left_eye_width', 'right_eye_width', 'left_eyebrow', 'nose_width',
             'right_eyebrow', 'nose_height', 'jaw_height', 'mouth_width', 'philtrum']]
        new_df = new_df.values.tolist()[0]

    dif = []
    for x in range(10):
        dif.append(round((1 - abs(rich_mean[x] - new_df[x]) / rich_mean[x]) * 100, 2))
    print('dif :', dif)
    print(np.mean(dif))

    # plt.ylabel()


    cat = ['eye_between', 'left_eye_width', 'right_eye_width', 'left_eyebrow', 'nose_width',
           'right_eyebrow', 'nose_height', 'jaw_height', 'mouth_width', 'philtrum']
    values = new_df

    N = len(cat)

    x_as = [n / float(N) * 2 * pi for n in range(N)]

    # Because our chart will be circular we need to append a copy of the first
    # value of each list at the end of each list with data
    values += values[:1]
    x_as += x_as[:1]
    print(x_as)
    # Set color of axes
    plt.rc('axes', linewidth=0.5, edgecolor="#888888")

    # Font
    fm.get_fontconfig_fonts()
    # font_location = '/usr/share/fonts/truetype/nanum/NanumGothicOTF.ttf'
    font_location = 'C:/windows/fonts/gaesung.ttf'  # For Windows
    font_name = fm.FontProperties(fname=font_location).get_name()
    plt.rc('font', family=font_name, size=10)

    # Create polar plot
    ax = plt.subplot(121, polar=True)

    # Set clockwise rotation. That is:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Set position of y-labels
    ax.set_rlabel_position(0)

    # Set color and linestyle of grid
    ax.xaxis.grid(True, color="#888888", linestyle='solid', linewidth=0.5)
    ax.yaxis.grid(True, color="#888888", linestyle='solid', linewidth=0.5)

    # Set number of radial axes and remove labels
    plt.xticks(x_as[:-1], [])

    # Set yticks
    plt.yticks([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5], ["0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.4"])

    # Plot data
    ax.plot(x_as, values, linewidth=2, linestyle='solid', zorder=3, marker='o', markersize=5)

    # Fill area
    ax.fill(x_as, values, color, alpha=0.3)

    # Set axes limits
    plt.ylim(0, 0.5)

    if percent==True:
        # plt.ylabel(f'{round(np.mean(dif), 4)} %', size=18, verticalalignment='top', rotation='horizontal',
        #            color='red', fontstyle='oblique')
        ax.set_title(f'{round(np.mean(dif), 4)} %', size=18,
                     color='red', rotation='horizontal')
        # plt.yticks([0], [f'{round(np.mean(dif), 4)} %'], color='red', size=18)


    # Draw ytick labels to make sure they fit properly
    for i in range(N):
           angle_rad = i / float(N) * 2 * pi
           if angle_rad == 0:
               ha, distance_ax = "center", 0.55
           elif 0 < angle_rad < pi:
               ha, distance_ax = "left", 0.5
           elif angle_rad == pi:
               ha, distance_ax = "center", 0.55
           else:
               ha, distance_ax = "right", 0.5

           ax.text(angle_rad, distance_ax, cat[i], size=12, horizontalalignment=ha, verticalalignment="center",
                   fontstyle='oblique')
    # if score == True:
    #     for i in range(N):
    #         angle_rad = i / float(N) * 2 * pi
    #         if angle_rad == 0:
    #             ha, distance_ax = "center", 0.6
    #         elif 0 < angle_rad < pi:
    #             ha, distance_ax = "left", 0.6
    #         elif angle_rad == pi:
    #             ha, distance_ax = "center", 0.6
    #         else:
    #             ha, distance_ax = "right", 0.6
    #
    #
    #         ax.text(angle_rad, distance_ax, f'{dif[i]} %',color='blue', size=10, horizontalalignment=ha, verticalalignment="center")

    # Show polar plot
    # fig.patch.set_visible(False)

    ax2 = plt.subplot(122)
    ax2.axis('off')
    ax2.axis('tight')
    clust_data = ['eye_between', 'left_eye_width', 'right_eye_width', 'left_eyebrow', 'nose_width',
                  'right_eyebrow', 'nose_height', 'jaw_height', 'mouth_width', 'philtrum']
    percent_data = dif

    collabel_1 = ['location']
    df = pd.DataFrame(clust_data, columns=collabel_1)
    df['percent(%)'] = percent_data
    print(df)
    ax2.table(cellText=df.values, colLabels=df.columns, loc='center',
              colLoc='center', cellLoc='center')

    if name_number == 1000:  # isinstance(name_number, list)
        pass
    else:
        name_number = name_number
        ax2.set_title(f'{return_type(name_number)}', size=18,  rotation='horizontal')





plot_radar_chart()  # 부자들의 평균을 그리는 함수 모두 기본값
# test = [0.357579, 0.192942, 0.172557, 0.222610, 0.222978,
#         0.295984, 0.250559, 0.258134, 0.391116, 0.137953]
# plot_radar_chart(test, color='yellow', percent=True)

plot_radar_chart(name_number=1010, color='yellow', percent=True)


# plt.tight_layout()
plt.show()

# 0 자수성가형, 1 금수저형, 2 투자의귀재, 3 또라이형, 4 자퇴형
# 5 결혼형, 6 시인형, 7 UN특사형, 8 정치인형, 9 professor type
# type = [0, 4, 4, 0, 4, 1, 0, 0, 0, 4, 1, 0, 0, 1, 1, 1, 5, 1, 5, 3,
#         0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 4, 0, 1, 0, 0, 1, 1, 5, 0, 0,
#         1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 5, 1, 0, 1, 1, 0, 1, 0, 0, 8,
#         0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 2, 1, 2, 0, 1, 0, 1, 2,
#         2, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 0, 1, 1, 1, 1, 4, 2, 0, 0,
#         0, 0, 2, 0, 5, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
#         4, 0, 1, 2, 2, 1, 6, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 1, 1, 4, 1, 0, 0, 4, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0,
#         2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0,
#         0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 5, 0, 0, 1,
#         0, 0, 1, 0, 1, 0, 4, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 1, 2, 1, 2, 1,
#         0, 0, 1, 1, 2, 0, 0, 1, 0, 1, 0, 2, 2, 4, 0, 5, 0, 0, 0, 0,
#         0, 0, 0, 2, 0, 0, 1, 0, 5, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1,
#         1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
#         0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 2, 1, 1, 1, 2, 0, 1, 2,
#         0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 2, 1, 0, 7, 1, 1, 0, 1,
#         1, 1, 1, 0, 0, 1, 1, 1, 0, 2, 2, 0, 0, 1, 1, 1, 0, 1, 0, 1,
#         0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 3, 1, 0, 3,
#         1, 1, 1, 0, 1, 1, 0, 0, 0, 3, 0, 0, 1, 1, 1, 1, 6, 0, 0, 1,
#         1, 0, 1, 1, 0, 0, 1, 2, 0, 1, 0, 0, 0, 0, 0, 1, 9, 1, 2, 2,
#         1, 1, 1, 0, 5, 0, 0, 4, 5, 0, 0, 1, 4, 3, 0, 4, 1, 1, 0, 0,
#         1, 3, 0, 0, 2, 2, 2, 0, 1, 1, 0, 4, 0, 1, 1, 1, 1, 1, 0, 1,
#         1, 3, 5, 0, 0, 0, 1, 1, 1, 0, 1, 5, 1, 0, 0, 6, 1, 4, 0, 2,
#         0, 1, 0, 0, 1, 1, 3, 0, 5, 1, 0, 0, 1, 1, 1, 1, 0, 1]

# 0 자수성가형, 1 금수저형, 2 투자의귀재, 3 또라이형, 4 자퇴형
# 5 결혼형, 6 시인형, 7 UN특사형, 8 정치인형, 9 professor type


