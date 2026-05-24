import matplotlib.pyplot as plt
import numpy as np

print("=========================================")
print("         📌 MATPLOTLIB VISUAL LAB        ")
print("=========================================\n")

# ----------------------------------------------------
# 1. Basic Plotting & Customization
# ----------------------------------------------------
# Task: Create a simple line plot of y = x^2 for x values from 0 to 10
x = np.linspace(0, 10, 100)
y = x**2

plt.figure(figsize=(6, 4))
# Customization: Color, line style, title, labels, legends, and grid line layout
plt.plot(x, y, color="darkorange", linestyle="-.", linewidth=2.5, label="y = x^2")
plt.title("Quadratic Growth Curve")
plt.xlabel("X Values")
plt.ylabel("Y Values (Squared)")
plt.grid(True, linestyle="--", alpha=0.6) 
plt.legend()
plt.savefig("line_plot.png") # Saving figures as images
plt.close()

# Task: Create a scatter plot of random points
x_rand = np.random.rand(50)
y_rand = np.random.rand(50)
plt.scatter(x_rand, y_rand, color="purple", marker="x", alpha=0.7)
plt.title("Random Vector Dispersion")
plt.savefig("scatter_plot.png")
plt.close()

# Task: Create a bar chart showing the population of different countries
countries = ["NZ", "Australia", "Fiji", "Samoa", "Tonga"]
populations = [5.3, 26.5, 0.9, 0.2, 0.1] # Stand-in population stats (Millions)
plt.bar(countries, populations, color="teal", edgecolor="black")
plt.title("Regional Population Estimates (Millions)")
plt.savefig("bar_chart.png")
plt.close()

# Task: Create a histogram of 1000 random numbers from a normal distribution
normal_data = np.random.randn(1000)
plt.hist(normal_data, bins=25, color="skyblue", edgecolor="grey")
plt.title("Normal Distribution Frequency Density")
plt.savefig("histogram.png")
plt.close()


# ----------------------------------------------------
# 2. Customization & Subplots (2x2 Grid)
# ----------------------------------------------------
# Task: Create a subplot with 2 rows and 2 columns, each with a different plot type
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Top-Left: Line
axes[0, 0].plot(x, y, color="red")
axes[0, 0].set_title("Line Curve")
axes[0, 0].grid(True)

# Top-Right: Scatter
axes[0, 1].scatter(x_rand, y_rand, color="blue")
axes[0, 1].set_title("Scatter Points")

# Bottom-Left: Bar
axes[1, 0].bar(countries, populations, color="green")
axes[1, 0].set_title("Population Distribution")

# Bottom-Right: Histogram
axes[1, 1].hist(normal_data, bins=15, color="yellow", edgecolor="black")
axes[1, 1].set_title("Data Histogram")

plt.tight_layout() # Optimizes formatting spacing
plt.savefig("combined_subplots.png")
plt.close()


# ----------------------------------------------------
# 3. Advanced Plotting
# ----------------------------------------------------
# Task: Create a pie chart showing the market share of different companies
companies = ["Apple", "Samsung", "Google", "Huawei", "Others"]
shares = [35, 30, 15, 10, 10]
plt.pie(shares, labels=companies, autopct="%1.1f%%", colors=["#ff9999","#66b3ff","#99ff99","#ffcc99","#c2c2f0"])
plt.title("Smartphone Market Share")
plt.savefig("pie_chart.png")
plt.close()

# Task: Create a box plot to visualize the distribution of a dataset
dataset = [np.random.normal(0, std, 100) for std in range(1, 4)]
plt.boxplot(dataset, patch_artist=True)
plt.title("Dataset Structural Variance (Box Plot)")
plt.savefig("box_plot.png")
plt.close()

# Task: Create a heatmap to visualize data in a 2D grid
heatmap_data = np.random.rand(10, 10)
plt.imshow(heatmap_data, cmap="viridis", interpolation="nearest")
plt.colorbar(label="Intensity Scale")
plt.title("2D Thermal Density Grid")
plt.savefig("heatmap.png")
plt.close()

# Task: Create a 3D plot of the function z = x^2 + y^2 using mplot3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x_3d = np.linspace(-5, 5, 50)
y_3d = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x_3d, y_3d)
Z = X**2 + Y**2

ax.plot_surface(X, Y, Z, cmap="magma")
ax.set_title("3D Paraboloid Space: z = x^2 + y^2")
plt.savefig("3d_plot.png")
plt.close()

print("\n🎉 Success! All 8 individual graphical files have generated as image assets.")