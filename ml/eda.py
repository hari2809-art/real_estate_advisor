import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("../data/india_housing_prices.csv")

print("Dataset Shape:", df.shape)
print("\nColumns:\n", df.columns)

# =========================
# BASIC INFO
# =========================
print("\n--- INFO ---")
print(df.info())

print("\n--- DESCRIPTION ---")
print(df.describe())

# =========================
# HANDLE MISSING VALUES (CHECK)
# =========================
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# =========================
# FEATURE ENGINEERING (FOR EDA)
# =========================
df['Price_per_SqFt'] = df['Price_in_Lakhs'] * 100000 / df['Size_in_SqFt']

# =========================
# 1. PRICE DISTRIBUTION
# =========================
plt.figure()
sns.histplot(df['Price_in_Lakhs'], kde=True)
plt.title("Price Distribution")
plt.xlabel("Price (Lakhs)")
plt.ylabel("Frequency")
plt.show()

# =========================
# 2. SIZE DISTRIBUTION
# =========================
plt.figure()
sns.histplot(df['Size_in_SqFt'], kde=True)
plt.title("Size Distribution")
plt.xlabel("Size (SqFt)")
plt.show()

# =========================
# 3. PRICE VS SIZE
# =========================
plt.figure()
sns.scatterplot(x=df['Size_in_SqFt'], y=df['Price_in_Lakhs'])
plt.title("Size vs Price")
plt.xlabel("Size")
plt.ylabel("Price")
plt.show()

# =========================
# 4. PRICE PER SQFT BY PROPERTY TYPE
# =========================
plt.figure()
sns.boxplot(x=df['Property_Type'], y=df['Price_per_SqFt'])
plt.title("Price per SqFt by Property Type")
plt.xticks(rotation=45)
plt.show()

# =========================
# 5. OUTLIER DETECTION
# =========================
plt.figure()
sns.boxplot(y=df['Price_per_SqFt'])
plt.title("Outliers in Price per SqFt")
plt.show()

# =========================
# 6. CITY-WISE PRICE
# =========================
plt.figure()
df.groupby('City')['Price_in_Lakhs'].mean().sort_values().plot(kind='barh')
plt.title("Average Price by City")
plt.xlabel("Price (Lakhs)")
plt.show()

# =========================
# 7. STATE-WISE PRICE PER SQFT
# =========================
plt.figure()
df.groupby('State')['Price_per_SqFt'].mean().sort_values().plot(kind='barh')
plt.title("State-wise Price per SqFt")
plt.show()

# =========================
# 8. BHK DISTRIBUTION
# =========================
plt.figure()
df['BHK'].value_counts().plot(kind='bar')
plt.title("BHK Distribution")
plt.xlabel("BHK")
plt.ylabel("Count")
plt.show()

# =========================
# 9. FURNISHED STATUS VS PRICE
# =========================
plt.figure()
sns.boxplot(x=df['Furnished_Status'], y=df['Price_in_Lakhs'])
plt.title("Price vs Furnished Status")
plt.xticks(rotation=45)
plt.show()

# =========================
# 10. NEARBY SCHOOLS VS PRICE
# =========================
plt.figure()
sns.scatterplot(x=df['Nearby_Schools'], y=df['Price_per_SqFt'])
plt.title("Schools vs Price per SqFt")
plt.show()

# =========================
# 11. NEARBY HOSPITALS VS PRICE
# =========================
plt.figure()
sns.scatterplot(x=df['Nearby_Hospitals'], y=df['Price_per_SqFt'])
plt.title("Hospitals vs Price per SqFt")
plt.show()

# =========================
# 12. PARKING VS PRICE
# =========================
plt.figure()
sns.boxplot(x=df['Parking_Space'], y=df['Price_in_Lakhs'])
plt.title("Parking vs Price")
plt.show()

# =========================
# 13. AMENITIES VS PRICE
# =========================
plt.figure()
sns.boxplot(x=df['Amenities'], y=df['Price_per_SqFt'])
plt.title("Amenities vs Price per SqFt")
plt.xticks(rotation=45)
plt.show()

# =========================
# 14. TRANSPORT ACCESS VS PRICE
# =========================
plt.figure()
sns.scatterplot(x=df['Public_Transport_Accessibility'], y=df['Price_per_SqFt'])
plt.title("Transport Access vs Price per SqFt")
plt.show()

# =========================
# 15. CORRELATION HEATMAP
# =========================
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# =========================
# SUMMARY
# =========================
print("\nEDA COMPLETED SUCCESSFULLY ✅")