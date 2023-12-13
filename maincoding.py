# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Functions to read and transpose data
def read_data_excel(excel_url, sheet_name, new_cols, countries):
    """
    Reads data from an Excel file and performs necessary preprocessing.

    Parameters:
    - excel_url (str): URL of the Excel file.
    - sheet_name (str): Name of the sheet containing data.
    - new_cols (list): List of columns to select from the data.
    - countries (list): List of countries to include in the analysis.

    Returns:
    - data_read (DataFrame): Preprocessed data.
    - data_transpose (DataFrame): Transposed data.
    """
    data_read = pd.read_excel(excel_url, sheet_name=sheet_name, skiprows=3)
    data_read = data_read[new_cols]
    data_read.set_index('Country Name', inplace=True)
    data_read = data_read.loc[countries]

    return data_read, data_read.T

# Parameters for reading and transposing data
sheet_name = 'Data'
new_cols = ['Country Name', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013']
countries = ['United Kingdom', 'United States', 'Afghanistan', 'Italy', 'China', 'Japan', 'Germany']

# The Excel URL below indicates GDP growth (annual %) for selected countries
excel_url_gdp = 'https://api.worldbank.org/v2/en/indicator/NY.GDP.MKTP.KD.ZG?downloadformat=excel'
data_gdp_read, data_GDP_transpose = read_data_excel(excel_url_gdp, sheet_name, new_cols, countries)

# The Excel URL below indicates Agriculture, forestry, and fishing, value added (% of GDP)
excel_url_agriculture = 'https://api.worldbank.org/v2/en/indicator/NV.AGR.TOTL.ZS?downloadformat=excel'
data_agriculture_read, data_agriculture_transpose = read_data_excel(excel_url_agriculture, sheet_name, new_cols, countries)

# The Excel URL below indicates electricity production from oil, gas, and coal sources (% of total)
excel_url_electricity = 'https://api.worldbank.org/v2/en/indicator/EG.ELC.FOSL.ZS?downloadformat=excel'
data_electricity_read, data_electricity_transpose = read_data_excel(excel_url_electricity, sheet_name, new_cols, countries)

# The Excel URL below indicates CO2 emissions (metric tons per capita)
excel_url_CO2 = 'https://api.worldbank.org/v2/en/indicator/EN.ATM.CO2E.PC?downloadformat=excel'
data_CO2, data_CO2_transpose = read_data_excel(excel_url_CO2, sheet_name, new_cols, countries)

# The Excel URL below indicates Forest area (% of land area)
excel_url_forest_area = 'https://api.worldbank.org/v2/en/indicator/AG.LND.FRST.ZS?downloadformat=excel'
data_forest_area, data_forest_area_transpose = read_data_excel(excel_url_forest_area, sheet_name, new_cols, countries)

# The Excel URL below indicates Arable land (% of land area)
excel_url_arable_land = 'https://api.worldbank.org/v2/en/indicator/AG.LND.ARBL.ZS?downloadformat=excel'
data_arable_land, data_arable_land_transpose = read_data_excel(excel_url_arable_land, sheet_name, new_cols, countries)

# The Excel URL below indicates Urban population growth (annual %)
excel_url_urban = 'https://api.worldbank.org/v2/en/indicator/SP.URB.GROW?downloadformat=excel'
data_urban_read, data_urban_transpose = read_data_excel(excel_url_urban, sheet_name, new_cols, countries)

# Print the transposed data
print(data_GDP_transpose)

# Describe the statistics of GDP growth (annual %)
GDP_statistics = data_GDP_transpose.describe()
print(GDP_statistics)

def multiple_plot(x_data, y_data, xlabel, ylabel, title, labels, colors):
    """
    Plots multiple line plots for the given data.

    Parameters:
    - x_data (array): X-axis data.
    - y_data (list): List of Y-axis data.
    - xlabel (str): X-axis label.
    - ylabel (str): Y-axis label.
    - title (str): Plot title.
    - labels (list): List of labels for each line.
    - colors (list): List of colors for each line.
    """
    plt.figure(figsize=(8, 6))
    plt.title(title, fontsize=10)
    for i in range(len(y_data)):
        plt.plot(x_data, y_data[i], label=labels[i], color=colors[i])
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.legend(bbox_to_anchor=(1.02, 1))
    plt.show()
    return

# Define a function to create a correlation heatmap
def correlation_heatmap(data, corr, title):
    """
    Displays a correlation heatmap for the given data.

    Parameters:
    - data (DataFrame): Input data.
    - corr (DataFrame): Correlation matrix.
    - title (str): Title for the heatmap.
    """
    plt.figure(figsize=(8, 6), dpi=200)
    plt.imshow(corr, cmap='plasma', interpolation='none')
    plt.colorbar()

    # Show all ticks and label them with the dataframe column name
    plt.xticks(range(len(data.columns)), data.columns, rotation=90, fontsize=10)
    plt.yticks(range(len(data.columns)), data.columns, rotation=0, fontsize=10)

    plt.title(title, fontsize=10)

    # Loop over data dimensions and create text annotations
    labels = corr.values
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            plt.text(j, i, '{:.2f}'.format(labels[i, j]),
                     ha="center", va="center", color="white")

    plt.show()

# Define the function to construct a multiple bar plot
def bar_plot(labels_array, width, y_data, y_label, label, title, rotation=0):
    """
    Plot a grouped bar plot.

    Parameters:
    - labels_array (array-like): X-axis labels.
    - width (float): Width of each bar group.
    - y_data (list of array-like): Y-axis data for each bar.
    - y_label (str): Y-axis label.
    - label (list): Labels for each bar group.
    - title (str): Plot title.
    - rotation (float): Rotation angle for X-axis labels.
    """
    x = np.arange(len(labels_array))
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

    for i in range(len(y_data)):
        plt.bar(x + width * i, y_data[i], width, label=label[i])

    plt.title(title, fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    plt.xlabel(None)
    plt.xticks(x + width * (len(y_data) - 1) / 2, labels_array, rotation=rotation)

    plt.legend()
    ax.tick_params(bottom=False, left=True)

    plt.show()

# Define the function to construct a Scator Plot plot
def scatter_plot(x_data, y_data, xlabel, ylabel, title, color='blue', size=8):
    """
    Plots a scatter plot for the given data.

    Parameters:
    - x_data (array): X-axis data.
    - y_data (array): Y-axis data.
    - xlabel (str): X-axis label.
    - ylabel (str): Y-axis label.
    - title (str): Plot title.
    - color (str): Color of the scatter points (default is blue).
    - size (int): Size of the points in the scatter plot (default is 8).
    """
    plt.figure(figsize=(8, 6), dpi=200)
    plt.scatter(x_data, y_data, color=color, s=size)
    plt.title(title, fontsize=10)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.show()

# Plot a multiple line plot for GDP growth (annual %)
x_data = data_GDP_transpose.index
y_data = [data_GDP_transpose['United Kingdom'],
          data_GDP_transpose['United States'],
          data_GDP_transpose['Afghanistan'],
          data_GDP_transpose['Italy'],
          data_GDP_transpose['China'],
          data_GDP_transpose['Japan'],
          data_GDP_transpose['Germany']]
xlabel = 'Years'
ylabel = '(%) GDP Growth'
labels = ['Uk', 'USA', 'AFG', 'Italy', 'China', 'Japan', 'Germany']
colors = ['purple', 'magenta', 'blue', 'green', 'yellow', 'red', 'black']
title = 'Annual (%) GDP Growth Countries'

# Plot the line plots for GDP of Countries
multiple_plot(x_data, y_data, xlabel, ylabel, title, labels, colors)

# Display the transposed data for CO2 emissions and GDP
print("CO2 Emissions:")
print(data_CO2_transpose)

print("GDP Growth:")
print(data_GDP_transpose)

# Merge CO2 and GDP data on the 'Country Name' index
merged_data = pd.merge(data_CO2_transpose, data_GDP_transpose, left_index=True, right_index=True, suffixes=('_CO2', '_GDP'))
# Calculate the correlation matrix for merged_data
correlation_matrix = merged_data.corr()
# Set up the matplotlib figure
plt.figure(figsize=(10, 8), dpi=200)
# Create a heatmap using seaborn
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
# Set the title
plt.title('Correlation Heatmap: CO2 Production vs. GDP (Energy Efficiency)', fontsize=10)

# Show the plot
plt.show()

# Display the preprocessed data
print(data_electricity_read)
print(data_electricity_transpose)

# Parameters for plotting multiple bar plots of electricity production from oil, gas and coal (% of total)
labels_array_agr = data_electricity_transpose.index
y_data_agr = [data_electricity_transpose['United Kingdom'], 
          data_electricity_transpose['United States'], 
          data_electricity_transpose['Afghanistan'],
          data_electricity_transpose['Italy'], 
          data_electricity_transpose['China'], 
          data_electricity_transpose['Japan']]
width_agr = 0.2
xlabel = 'Year'
y_label_agr = '% electricity production'
label_agr = ['UK', 'USA', 'Afghanistan', 'Italy', 'China', 'Japan', 'Germany']
colors = ['red', 'magenta', 'blue', 'yellow', 'green', 'purple', 'black']
title_agr = 'Electricity production from oil, gas and coal sources (% of total)'

# Call the function to plot the multiple bar plot
bar_plot(labels_array_agr, width_agr, y_data_agr, y_label_agr, label_agr, title_agr, rotation=55)

# Create a dataframe for United States of America using selected indicators
data_USA = {
    'Urban pop. growth': data_urban_transpose['United States'],
    'Electricity production': data_electricity_transpose['United States'],
    'Agric. forestry and Fisheries': data_agriculture_transpose['United States'],
    'CO2 Emissions': data_CO2_transpose['United States'],
    'Forest Area': data_forest_area_transpose['United States'],
    'GDP Annual Growth': data_GDP_transpose['United States']
}
df_USA = pd.DataFrame(data_USA)

# Display the dataframe and correlation matrix
print(df_USA)
corr_USA = df_USA.corr()
print(corr_USA)

# Display the correlation heatmap for USA
correlation_heatmap(df_USA, corr_USA, 'United States')

# Create a dataframe for Germany using selected indicators
data_Italy = {
    'Urban pop. growth': data_urban_transpose['Italy'],
    'Electricity production': data_electricity_transpose['Italy'],
    'Agric. forestry and Fisheries': data_agriculture_transpose['Italy'],
    'CO2 Emissions': data_CO2_transpose['Italy'],
    'Forest Area': data_forest_area_transpose['Italy'],
    'GDP Annual Growth': data_GDP_transpose['Italy']
}
df_Italy = pd.DataFrame(data_Italy)

# Display the dataframe and correlation matrix
print(df_Italy)
corr_Italy = df_Italy.corr()
print(corr_Italy)

# Display the correlation heatmap for Germany
correlation_heatmap(df_Italy, corr_Italy, 'Italy')

# Plot a grouped bar plot for Agriculture, forestry, and fishing, value added (% of GDP) for countries
labels_array_agr = countries
width_agr = 0.2
y_data_agr = [
    data_agriculture_read['2009'],
    data_agriculture_read['2010'],
    data_agriculture_read['2011'],
    data_agriculture_read['2012']
]
y_label_agr = '% of GDP'
label_agr = ['Year 2009', 'Year 2010', 'Year 2011', 'Year 2012']
title_agr = 'Agriculture, forestry, and fishing, value added (% of GDP) for Countries'

# Plot the grouped bar plot forcountries
bar_plot(labels_array_agr, width_agr, y_data_agr, y_label_agr, label_agr, title_agr, rotation=55)

# Plot a multiple line plot for Forest Area (% of land area) for selected countries
x_data_forest = data_forest_area_transpose.index
y_data_forest = [
    data_forest_area_transpose['United Kingdom'],
    data_forest_area_transpose['United States'],
    data_forest_area_transpose['Afghanistan'],
    data_forest_area_transpose['Italy'],
    data_forest_area_transpose['China'],
    data_forest_area_transpose['Japan'],
    data_forest_area_transpose['Germany']
]
xlabel_forest = 'Years'
ylabel_forest = '(%) Forest Area'
labels_forest = ['UK', 'USA', 'AFG', 'Italy', 'China', 'Japan', 'Germany']
colors_forest = ['purple', 'magenta', 'blue', 'green', 'yellow', 'red', 'black']
title_forest = 'Annual (%) Forest Area for Selected Countries'

# Plot the line plots for Forest Area of selected countries
multiple_plot(x_data_forest, y_data_forest, xlabel_forest, ylabel_forest, title_forest, labels_forest, colors_forest)

# Merge electricity production and GDP data on the 'Country Name' index
merged_electricity_gdp = pd.merge(data_electricity_transpose, data_GDP_transpose, left_index=True, right_index=True, suffixes=('_Electricity', '_GDP'))

# Calculate the correlation matrix for merged_electricity_gdp
correlation_matrix_electricity_gdp = merged_electricity_gdp.corr()

# Display the correlation matrix
print("Correlation Matrix (Electricity Production vs. GDP Growth):")
print(correlation_matrix_electricity_gdp)

# Display the results in tabular form
correlation_table = pd.DataFrame(correlation_matrix_electricity_gdp['China_GDP'])
correlation_table.columns = ['Correlation with GDP Growth']
print("\nCorrelation with GDP Growth for China:")
print(correlation_table)

# Example usage for the scatter plot
scatter_plot(merged_electricity_gdp['China_Electricity'], merged_electricity_gdp['China_GDP'],
             'Electricity Production (% of total)', 'GDP Growth (%)',
             'Scatter Plot: Electricity Production vs. GDP Growth (China)')
