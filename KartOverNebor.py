# importing modules and packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.image as mpimg
from matplotlib.patches import Patch


colors = ['green', 'yellow', 'orange', 'red', 'darkred']
labels = [
    'Under 1300 mm',
    '1300–1699 mm',
    '1700–2499 mm',
    '2500–3199 mm',
    '3200 mm eller mer'
]
patches = [Patch(color=colors[i], label=labels[i]) for i in range(len(colors))]


def index_from_nedbor(x):
    if x < 1300: return 0
    if x < 1700: return 1
    if x < 2500: return 2
    if x < 3200: return 3
    return 4

def color_from_nedbor(nedbor):
    return colors[index_from_nedbor(nedbor)]

def size_from_nedbor(nedbor):
    return 350

def label_from_nedbor(nedbor):
    return str(int(nedbor / 100))

def draw_label_and_ticks():
    xlabels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    axGraph.set_xticks(np.linspace(1, 12, 12))
    axGraph.set_xticklabels(xlabels)

def draw_the_map():
    axMap.cla()
    axMap.imshow(img, extent=(0, 13, 0, 10))
    df_year = df.groupby(['X', 'Y']).agg({'Nedbor': 'sum'}).reset_index()
    xr = df_year['X'].tolist()
    yr = df_year['Y'].tolist()
    nedborAar = df_year['Nedbor']
    ColorList = [color_from_nedbor(n) for n in nedborAar]
    axMap.scatter(xr, yr, c=ColorList, s=size_from_nedbor(nedborAar / 12), alpha=1)
    labels_txt = [label_from_nedbor(n) for n in nedborAar]
    for i, y in enumerate(xr):
        axMap.text(xr[i], yr[i], s=labels_txt[i], color='black', fontsize=8, ha='center', va='center')

def on_click(event):
    global marked_point
    if event.inaxes != axMap:
        return
    marked_point = (event.xdata, event.ydata)
    x, y = marked_point

    vectors = []
    months = np.linspace(1, 12, 12)
    for mnd in months:
        vectors.append([x, y, mnd])
    AtPoint = np.vstack(vectors)

    AtPointM = poly.fit_transform(AtPoint)
    y_pred = model.predict(AtPointM)
    aarsnedbor = sum(y_pred)

    axGraph.cla()
    draw_the_map()
    axMap.set_title(f"C: ({x:.1f},{y:.1f}) - klikk rød er estimert")

    axMap.text(x, y, s=label_from_nedbor(aarsnedbor), color='black', fontsize=8, ha='center', va='center')
    axGraph.set_title(f"Nedbør per måned, Årsnedbør {int(aarsnedbor)} mm")

    colorsPred = [color_from_nedbor(nedbor * 12) for nedbor in y_pred]
    axMap.scatter(x, y, c="black", s=size_from_nedbor(aarsnedbor) * 1.5, marker="o")
    axMap.scatter(x, y, c=color_from_nedbor(aarsnedbor), s=size_from_nedbor(aarsnedbor), marker="o")
    axGraph.bar(months, y_pred, color=colorsPred)

    # 🟣 Legg til lilla strek for gjennomsnittlig månedlig nedbør
    gjennomsnitt = np.mean(y_pred)
    axGraph.axhline(y=gjennomsnitt, color='red', linestyle='--', linewidth=2, label=f"Gjennomsnitt: {gjennomsnitt:.1f} mm")
    axGraph.legend(loc='upper left', fontsize=8)

    draw_label_and_ticks()
    plt.draw()

df = pd.read_csv('NedborX.csv')
img = mpimg.imread('Bergen.png')
marked_point = (0, 0)

ns = df['Nedbor']
X = df.drop('Nedbor', axis=1)
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X_poly, ns, test_size=0.25)

model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

r_squared = r2_score(Y_test, Y_pred)
print(f"R-squared: {r_squared:.2f}")
print('mean_absolute_error (mnd): ', mean_absolute_error(Y_test, Y_pred))


fig = plt.figure(figsize=(12, 5))


axLegend = fig.add_axes((0.05, 0.1, 0.2, 0.8))
axLegend.legend(handles=patches, loc='center', fontsize=10, frameon=False)
axLegend.axis('off')
axLegend.set_title("Årsnedbør (mm)", fontsize=11)


axGraph = fig.add_axes((0.3, 0.1, 0.3, 0.8))
axMap = fig.add_axes((0.63, 0.1, 0.35, 0.8))

draw_label_and_ticks()
draw_the_map()
axMap.set_title("Årsnedbør Stor-Bergen")
axGraph.set_title("Per måned")
axMap.axis('off')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
axMap.margins(x=0.01, y=0.01)

plt.connect('button_press_event', on_click)
plt.show()

