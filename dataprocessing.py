# Import

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Reading Files

essays = pd.read_csv('kdd-cup-2014-predicting-excitement-at-donors-choose/data/essays.csv')
outcomes = pd.read_csv('kdd-cup-2014-predicting-excitement-at-donors-choose/data/outcomes.csv')
projects = pd.read_csv('kdd-cup-2014-predicting-excitement-at-donors-choose/data/projects.csv')
donations = pd.read_csv('kdd-cup-2014-predicting-excitement-at-donors-choose/data/donations.csv')

# Merging

combined = projects.merge(outcomes,how='inner',on='projectid').merge(essays,how='inner',on='projectid')

# Preparing the Data

#dropping columns that are not important such as teacher account id
#dropping columns with too many categories such as school city

columns = ['projectid','school_state','school_metro','school_magnet','school_nlns',
           'school_kipp','school_charter','school_charter_ready_promise','teacher_teach_for_america',
           'teacher_ny_teaching_fellow','primary_focus_subject','primary_focus_area',
           'resource_type','poverty_level','grade_level','total_price_excluding_optional_support',
           'total_price_including_optional_support','students_reached','eligible_double_your_impact_match',
           'eligible_almost_home_match','date_posted','fully_funded','short_description','need_statement','essay']

data = combined[columns]

# Mapping State to Regions

state_to_region = {
    # Northeast
    'CT': 'Northeast', 'ME': 'Northeast', 'MA': 'Northeast', 'NH': 'Northeast',
    'RI': 'Northeast', 'VT': 'Northeast', 'NJ': 'Northeast', 'NY': 'Northeast', 'PA': 'Northeast',
    
    # Midwest
    'IL': 'Midwest', 'IN': 'Midwest', 'MI': 'Midwest', 'OH': 'Midwest', 'WI': 'Midwest',
    'IA': 'Midwest', 'KS': 'Midwest', 'MN': 'Midwest', 'MO': 'Midwest', 'NE': 'Midwest', 'ND': 'Midwest', 'SD': 'Midwest',
    
    # South
    'DE': 'South', 'FL': 'South', 'GA': 'South', 'MD': 'South', 'NC': 'South', 'SC': 'South', 'VA': 'South', 'DC': 'South',
    'WV': 'South', 'AL': 'South', 'KY': 'South', 'MS': 'South', 'TN': 'South', 'AR': 'South', 'LA': 'South',
    'OK': 'South', 'TX': 'South',
    
    # West
    'AZ': 'West', 'CO': 'West', 'ID': 'West', 'MT': 'West', 'NV': 'West', 'NM': 'West',
    'UT': 'West', 'WY': 'West', 'AK': 'West', 'CA': 'West', 'HI': 'West', 'OR': 'West', 'WA': 'West'
}

data['school_state_region'] = data['school_state'].map(state_to_region)

#convert t/f values for outcomes to binary

binary_map = {'f':0,'t':1}
data['fully_funded'] = data['fully_funded'].map(binary_map)

# columns_binary = ['school_magnet','school_nlns',
#            'school_kipp','school_charter','school_charter_ready_promise','teacher_teach_for_america',
#            'teacher_ny_teaching_fellow','eligible_double_your_impact_match',
#            'eligible_almost_home_match','fully_funded']

# binary_map = {'f':0,'t':1}

# for col in columns_binary:
#     data.loc[:,col] = data[col].map(binary_map)

#Sort ascending chronologically

data = data.sort_values(by='date_posted')

#Drop missing value

data.dropna(inplace=True)

# Addressing Outlier

q = data['students_reached'].quantile(0.95)

scaler = StandardScaler()
data_noutliers = data[data['students_reached'] < q]
data_noutliers = data_noutliers[data_noutliers['students_reached'] > 0.0]

data_noutliers['students_reached_scaled'] = scaler.fit_transform(data_noutliers.loc[:,['students_reached']])


q = data['total_price_excluding_optional_support'].quantile(0.95)

data_noutliers = data[data['total_price_excluding_optional_support'] < q]
data_noutliers = data_noutliers[data_noutliers['total_price_excluding_optional_support'] > 0.0]

data_noutliers['total_price_excluding_optional_support_scaled'] = scaler.fit_transform(data_noutliers.loc[:,['total_price_excluding_optional_support']])

data_noutliers.reset_index(drop=True,inplace=True)
data.reset_index(drop=True,inplace=True)

# print(data)
# print(data_noutliers)