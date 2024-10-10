# prompt: welcome user to the lab report generator and ask them to input the path to the experimental data set and print the first few rows

import pandas as pd
import streamlit as st
print("Welcome to the Lab Report Generator!")

# Ask the user to upload their file
uploaded_file  = st.file_uploader("Upload your experimental data set:")

if uploaded_file is not None:
    # Get the file path from the uploaded file
    file_path = uploaded_file.name

try:
  df = pd.read_csv(file_path)
  print("\nHere are the first few rows of your data:")
  print(df.head())
except FileNotFoundError:
  print("Error: File not found. Please check the file path and try again.")
except Exception as e:
  print(f"An error occurred: {e}")
# prompt: remove missing values in data after showing user and asking then show again after replacing with mean

if input("Do you want to handle missing values? (yes/no): ").lower() == 'yes':
  print("\nHere's a summary of missing values in your data:")
  print(df.isnull().sum())

  if df.isnull().values.any():
    print("\nMissing values detected.")
    if input("Do you want to remove rows with missing values? (yes/no): ").lower() == 'yes':
      df.dropna(inplace=True)
      print("\nRows with missing values have been removed.")
      print("\nHere's a summary of missing values after removal:")
      print(df.isnull().sum())

    if input("Do you want to replace missing values with the mean of their respective columns? (yes/no): ").lower() == 'yes':
      for column in df.columns:
        if df[column].isnull().any():
          df[column].fillna(df[column].mean(), inplace=True)
      print("\nMissing values have been replaced with the mean of their respective columns.")
      print("\nHere's a summary of missing values after replacement:")
      print(df.isnull().sum())

  else:
    print("\nNo missing values detected.")
# prompt: remove outliers in data after listing to user outliers in each column in one list including number of outliers in each column and asking if they want to remove them;  show data statistics before and after removing only outliers in each column

if input("Do you want to handle outliers? (yes/no): ").lower() == 'yes':
  outlier_info = []
  for column in df.select_dtypes(include=['number']).columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    num_outliers = len(df[(df[column] < lower_bound) | (df[column] > upper_bound)])
    if num_outliers > 0:
        outlier_info.append((column, num_outliers))

  if outlier_info:
      print("\nPotential outliers detected in the following columns:")
      for column, num_outliers in outlier_info:
          print(f"- {column}: {num_outliers} outliers")

      if input("Do you want to remove outliers? (yes/no): ").lower() == 'yes':
          print("\nData statistics before removing outliers:")
          print(df.describe())

          for column, _ in outlier_info:
              Q1 = df[column].quantile(0.25)
              Q3 = df[column].quantile(0.75)
              IQR = Q3 - Q1
              lower_bound = Q1 - 1.5 * IQR
              upper_bound = Q3 + 1.5 * IQR
              df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

          print("\nData statistics after removing outliers:")
          print(df.describe())
      else:
          print("\nOutliers will not be removed.")
  else:
      print("\nNo potential outliers detected.")
# prompt: ask user if they want to use one hot encoding on their data, explain they it converts categorical to numerical for analysis if they need an explaination, if they decide yes one hot encode

import pandas as pd

if input("Do you need an explanation of one-hot encoding? (yes/no): ").lower() == 'yes':
  print("\nOne-hot encoding is a process used to convert categorical data (data represented as labels or text) into numerical data that can be used for analysis by machine learning algorithms.")
  print("It creates new binary columns for each unique category in the original column, representing whether a specific row belongs to that category or not.")
  print("For example, if you have a column 'Color' with categories 'Red', 'Blue', and 'Green', one-hot encoding will create three new columns: 'Color_Red', 'Color_Blue', and 'Color_Green'.")
  print("Each row will have a 1 in the column corresponding to its color and 0s in the other columns.")
if input("Do you want to use one-hot encoding on your data? (yes/no): ").lower() == 'yes':
  categorical_columns = df.select_dtypes(include=['object']).columns
  if len(categorical_columns) > 0:
    print(f"\nThe following columns are categorical and can be one-hot encoded: {', '.join(categorical_columns)}")
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=False)
    print("\nOne-hot encoding has been applied successfully.")
    print(df.head())
  else:
    print("\nNo categorical columns were found in your data. One-hot encoding is not applicable.")
# prompt: if user wants to calculate conversion rate in chemistry between two columns the user picks from numbered list of columns, then print the two columns and the calculated column, ask the user columns that represent each variable in the equation

if input("Do you want to calculate conversion rate? (yes/no): ").lower() == 'yes':
  print("\nAvailable columns:")
  for i, col in enumerate(df.columns):
    print(f"{i+1}. {col}")

  while True:
    try:
      reactant_col_num = int(input("Enter the column number representing the initial moles of reactant: "))
      product_col_num = int(input("Enter the column number representing the final moles of product: "))

      if 0 <= reactant_col_num < len(df.columns) and 0 <= product_col_num < len(df.columns):
        break
      else:
        print("Invalid column number. Please try again.")
    except ValueError:
      print("Invalid input. Please enter a number.")

  reactant_col = df.columns[reactant_col_num]
  product_col = df.columns[product_col_num]

  df['Conversion Rate'] = (df[product_col] / df[reactant_col]) * 100

  print(f"\nHere are the columns {reactant_col}, {product_col} and the calculated Conversion Rate:")
  print(df[[reactant_col, product_col, 'Conversion Rate']])
# prompt: if user wants to calculate reaction efficiency in chemistry between two columns the user picks from numbered list of columns, then print the two columns and the calculated column, ask the user columns that represent each variable in the equation

if input("Do you want to calculate reaction efficiency? (yes/no): ").lower() == 'yes':
  print("\nAvailable columns:")
  for i, col in enumerate(df.columns):
    print(f"{i+1}. {col}")

  while True:
    try:
      theoretical_yield_col_num = int(input("Enter the column number representing the theoretical yield: "))
      actual_yield_col_num = int(input("Enter the column number representing the actual yield: "))

      if 0 <= theoretical_yield_col_num < len(df.columns) and 0 <= actual_yield_col_num < len(df.columns):
        break
      else:
        print("Invalid column number. Please try again.")
    except ValueError:
      print("Invalid input. Please enter a number.")

  theoretical_yield_col = df.columns[theoretical_yield_col_num]
  actual_yield_col = df.columns[actual_yield_col_num]

  df['Reaction Efficiency'] = (df[actual_yield_col] / df[theoretical_yield_col]) * 100

  print(f"\nHere are the columns {theoretical_yield_col}, {actual_yield_col} and the calculated Reaction Efficiency:")
  print(df[[theoretical_yield_col, actual_yield_col, 'Reaction Efficiency']])
# prompt: if user wants to calculate energy consumption in chemistry between two columns the user picks from numbered list of columns, then print the two columns and the calculated column, ask the user columns that represent each variable in the equation

# Energy consumption calculation
if input("Do you want to calculate energy consumption? (yes/no): ").lower() == 'yes':
  print("\nAvailable columns:")
  for i, col in enumerate(df.columns):
    print(f"{i+1}. {col}")

  while True:
    try:
      energy_input_col_num = int(input("Enter the column number representing the energy input: ")) - 1
      time_col_num = int(input("Enter the column number representing the time: ")) - 1

      if 0 <= energy_input_col_num < len(df.columns) and 0 <= time_col_num < len(df.columns):
        break
      else:
        print("Invalid column number. Please try again.")
    except ValueError:
      print("Invalid input. Please enter a number.")

  energy_input_col = df.columns[energy_input_col_num]
  time_col = df.columns[time_col_num]

  df['Energy Consumption'] = df[energy_input_col] * df[time_col]

  print(f"\nHere are the columns {energy_input_col}, {time_col} and the calculated Energy Consumption:")
  print(df[[energy_input_col, time_col, 'Energy Consumption']])
# Initialize an empty list to store the charts
charts_to_include = []  # Changed to a list
import matplotlib.pyplot as plt
import seaborn as sns  # Import seaborn
import io

# Initialize an empty list to store the charts
charts_to_include = []  # Changed to a list

while True:
    print("\nAvailable columns for box plots:")
    for i, col in enumerate(df.select_dtypes(include=['number']).columns):
        print(f"{i + 1}. {col}")

    col_num = input("Enter the column number to create a box plot (or 'done' to finish): ")

    if col_num.lower() == 'done':
        break

    try:
        col_num = int(col_num) - 1
        if 0 <= col_num < len(df.select_dtypes(include=['number']).columns):
            column_name = df.select_dtypes(include=['number']).columns[col_num]

            # Create the box plot
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=df[column_name])  # Create the box plot
            plt.title(f'Box Plot of {column_name}')
            plt.tight_layout()  # Adjust layout for better fit

            # Save the chart to a BytesIO object
            buf_boxplot = io.BytesIO()
            plt.savefig(buf_boxplot, format='png')
            plt.close()  # Close the plot to free memory
            buf_boxplot.seek(0)  # Move to the beginning of the BytesIO buffer

            # Append the BytesIO buffer to charts_to_include
            charts_to_include.append(buf_boxplot)  # Now append works since it's a list

            print(f"Box plot for {column_name} saved.")
        else:
            print("Invalid column number. Please try again.")
    except ValueError:
        print("Invalid input. Please enter a number or 'done'.")
import seaborn as sns
import io
import matplotlib.pyplot as plt

# Example of creating a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
# Save the chart to a BytesIO object
buf_heatmap = io.BytesIO()
plt.savefig(buf_heatmap, format='png')
buf_heatmap.seek(0)  # Move to the beginning of the BytesIO buffer
charts_to_include.append(buf_heatmap)  # Store in the list

plt.close()  # Close the plot to free memory
print(f"Heat map saved.")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io

# Assuming df is your DataFrame that contains the data
  # List to store charts for the report

while True:
    print("\nAvailable numerical columns for scatter plots:")
    for i, col in enumerate(df.select_dtypes(include=['number']).columns):
        print(f"{i + 1}. {col}")

    try:
        x_col_num = int(input("Enter the column number for the x-axis (or 0 to exit): ")) - 1
        if x_col_num == -1:
            break
        if 0 <= x_col_num < len(df.select_dtypes(include=['number']).columns):
            x_col = df.columns[x_col_num]
            y_col_num = int(input("Enter the column number for the y-axis: ")) - 1
            if 0 <= y_col_num < len(df.select_dtypes(include=['number']).columns):
                y_col = df.columns[y_col_num]

                # Create the scatter plot with regression line
                plt.figure(figsize=(6, 4))
                sns.regplot(x=df[x_col], y=df[y_col])  # Include regression line
                plt.title(f"Scatter Plot: {x_col} vs {y_col}")
                plt.xlabel(x_col)
                plt.ylabel(y_col)

                # Save the chart to a BytesIO object
                buf_scatterplot = io.BytesIO()
                plt.savefig(buf_scatterplot, format='png')
                buf_scatterplot.seek(0)  # Move to the beginning of the BytesIO buffer
                charts_to_include.append(buf_scatterplot)  # Store in the list

                plt.close()  # Close the plot to free memory
                print(f"Scatter plot for {x_col} vs {y_col} saved.")

            else:
                print("Invalid column number for y-axis. Please try again.")
        else:
            print("Invalid column number for x-axis. Please try again.")
    except ValueError:
        print("Invalid input. Please enter a number.")
# Install necessary packages


from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib import utils
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from PIL import Image
import pandas as pd
import io
import os

def wrap_text(text, font, max_width):
    """Wraps text to fit within the specified width."""
    words = text.split(' ')
    wrapped_lines = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        if pdfmetrics.stringWidth(test_line, font[0], font[1]) <= max_width:
            current_line = test_line
        else:
            wrapped_lines.append(current_line)
            current_line = word

    if current_line:
        wrapped_lines.append(current_line)

    return wrapped_lines

def create_lab_report_with_calculations(df, charts_to_include, filename="lab_report.pdf"):
    """Creates a PDF lab report with calculated columns and charts."""

    if not charts_to_include:
        print("No charts to include in the report.")
        return

    # Create a PDF canvas
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica", 16)
    c.drawString(1 * inch, height - 1 * inch, "Lab Report")

    # Section: Calculated Columns
    y_position = height - 2 * inch
    c.setFont("Helvetica", 14)
    c.drawString(1 * inch, y_position, "Calculated Data:")

    y_position -= 0.75 * inch
    calculated_cols = [col for col in df.columns if 'Rate' in col or 'Efficiency' in col or 'Consumption' in col]

    for col in calculated_cols:
        if y_position < 1 * inch:
            c.showPage()
            y_position = height - 1.5 * inch

        c.setFont("Helvetica", 12)
        wrapped_lines = wrap_text(f"{col}: {df[col].to_list()}", ("Helvetica", 12), width - 2 * inch)
        for line in wrapped_lines:
            c.drawString(1 * inch, y_position, line)
            y_position -= 0.4 * inch
            if y_position < 1 * inch:
                c.showPage()
                y_position = height - 1.5 * inch

    # Section: Summary Statistics
    if y_position < 1 * inch:
        c.showPage()
        y_position = height - 1.5 * inch

    c.setFont("Helvetica", 14)
    c.drawString(1 * inch, y_position, "Summary Statistics:")

    y_position -= 0.75 * inch
    summary_stats = df.describe()
    summary_str = summary_stats.to_string()

    # Print summary statistics with three columns per line
    for line in summary_str.split('\n'):
        if y_position < 1 * inch:
            c.showPage()
            y_position = height - 1.5 * inch

        c.setFont("Helvetica", 12)
        wrapped_lines = wrap_text(line, ("Helvetica", 12), width - 2 * inch)
        for wrapped_line in wrapped_lines:
            c.drawString(1 * inch, y_position, wrapped_line)
            y_position -= 0.4 * inch
            if y_position < 1 * inch:
                c.showPage()
                y_position = height - 1.5 * inch

    # Section: Highly Correlated Variables
    if y_position < 1 * inch:
        c.showPage()
        y_position = height - 1.5 * inch

    c.setFont("Helvetica", 14)
    c.drawString(1 * inch, y_position, "Highly Correlated Variables:")

    y_position -= 0.75 * inch
    corr_matrix = df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.8:
                if y_position < 1 * inch:
                    c.showPage()
                    y_position = height - 1.5 * inch

                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                c.setFont("Helvetica", 12)
                c.drawString(1 * inch, y_position, f"{col1} and {col2}: Correlation = {corr_value:.2f}")
                y_position -= 0.4 * inch

    # Section: Charts
    if y_position < 1 * inch:
        c.showPage()
        y_position = height - 1.5 * inch

    c.setFont("Helvetica", 14)
    c.drawString(1 * inch, y_position, "Charts:")
    y_position -= 0.75 * inch  # Move down for the charts

    # Iterate through the charts to include
    for i, chart_data in enumerate(charts_to_include):
        if y_position < 1 * inch:
            c.showPage()
            y_position = height - 1.5 * inch

        # Convert BytesIO to PIL Image and save to temporary file
        img = Image.open(chart_data)  # Open the BytesIO object directly
        temp_image_path = f"temp_chart_{i}.png"
        img.save(temp_image_path, format="PNG")  # Save the image as PNG

        # Draw the chart using the temporary file path
        c.drawImage(temp_image_path, 1 * inch, y_position - 3 * inch, width=5 * inch, height=3 * inch, mask='auto')
        y_position -= 4.5 * inch  # Move down for the next chart

        # Remove the temporary file
        import os
        os.remove(temp_image_path)  # Remove temporary files after use

    # Save the PDF
    c.save()
    print(f"{filename} created successfully.")

# Example usage
create_lab_report_with_calculations(df, charts_to_include)
