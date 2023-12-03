import pandas as pd

data = pd.read_csv('2019_kbo_for_kaggle_v2.csv')  # csv 파일 읽어옴
df = pd.DataFrame(data)  # DataFrame 생성

#2-1-(1)
year_ = {}  # 연도별 데이터 필터링

print("<10 players in hits>")
for year in range(2015, 2019):
    year_data = df[df['year'] == year]  # 연도별 데이터 수집
    hits = year_data[['batter_name', 'H']]  # 이름과 hits의 전체 데이터 저장
    # hits를 기준으로 내림차순 정렬
    hits_sorted = hits.sort_values(by=hits.columns[1], ascending=False).head(10)
    year_[year] = hits_sorted  # 결과 저장

hits_table = pd.concat(year_, keys=year_.keys())  # 결과를 표로 생성
print(hits_table)  # 결과 출력
print('\n')

year_ = {}  # year_ 딕셔너리 초기화

print("<10 players in batting average>")
for year in range(2015, 2019):
    year_data = df[df['year'] == year]
    avg = year_data[['batter_name', 'avg']]  # 이름과 batting average의 전체 데이터 저장
    # batting average를 기준으로 내림차순 정렬
    avg_sorted = avg.sort_values(by=avg.columns[1], ascending=False).head(10)
    year_[year] = avg_sorted  # 결과 저장

avg_table = pd.concat(year_, keys=year_.keys())  # 결과를 표로 생성
print(avg_table)  # 결과 출력
print('\n')

year_ = {}  # year_ 딕셔너리 초기화

print("<10 players in homerun>")
for year in range(2015, 2019):
    year_data = df[df['year'] == year]
    HR = year_data[['batter_name', 'HR']]  # 이름과 homerun 전체 데이터 저장
    # homerun을 기준으로 내림차순 정렬
    HR_sorted = HR.sort_values(by=HR.columns[1], ascending=False).head(10)
    year_[year] = HR_sorted  # 결과 저장

HR_table = pd.concat(year_, keys=year_.keys())  # 결과를 표로 생성
print(HR_table)  # 결과 출력
print('\n')

print("<10 players in on-based percentage>")
for year in range(2015, 2019):
    year_data = df[df['year'] == year]
    OBP = year_data[['batter_name', 'OBP']]  # 이름과 OBP 전체 데이터 저장
    # OBP를 기준으로 내림차순 정렬
    OBP_sorted = OBP.sort_values(by=OBP.columns[1], ascending=False).head(10)
    year_[year] = OBP_sorted  # 결과 저장

OBP_table = pd.concat(year_, keys=year_.keys())  # 결과를 표로 생성
print(OBP_table)  # 결과 출력
print('\n')

#2-1-(2)
def process_position_data(df, position):    # 조건에 맞는 데이터 필터링
    position_data = df[(df['year'] == 2018) & (df['cp'] == position)] # 2018년의 포지션별 데이터 불러오기
    position_data = position_data[['batter_name', 'cp', 'war']]
    # war을 기준으로 내림차순 정렬 & 첫번째 행 선택
    position_data_sorted = position_data.sort_values(by='war', ascending=False).head(1)
    return position_data_sorted

positions = ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']
all_data = pd.DataFrame() #모든 포지션 값을 저장해줄 DataFrame 생성
# 각 포지션에 대한 데이터 처리 및 저장
for position in positions:
    data_position = process_position_data(df, position)
    all_data = pd.concat([all_data, data_position])
print("<player with the highest war by position in 2018>")
print(all_data)
print('\n')

#2-1-(3)

#  row_data 들에 대하여 salary와의  correlation 계산
row_data = ['salary', 'R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG']
row_data = df[row_data]
correlations_all_data = row_data.corrwith(row_data['salary'])

print("<correlation with salary>")
print(correlations_all_data)

# salary를 제외한 가장 큰 상관계수를 가지는 변수 이름 찾기
max_corr_variable = correlations_all_data.iloc[1:].idxmax()

print("\n")
print("** highest correlation with salary is", max_corr_variable)
