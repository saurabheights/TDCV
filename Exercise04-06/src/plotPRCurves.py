from matplotlib import pyplot
import csv
from collections import defaultdict

filepath = '/media/sk/6a4a41c4-a920-46db-84c5-69e0450c2dd0/mega/TUM-Study/TrackingAndDetectionInComputerVision/Exercises/Exercise04-06/output/Trees-50_subsetPercent-50-undersampling_0-augment_1-strideX_2-strideY_2-NMS_MIN_0.1-NMS_Max_0.5-NMS_CONF_0.6/predictionRecallValues.csv';


columns = defaultdict(list) # each value in each column is appended to a list

with open(filepath) as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value 
            columns[k].append(v) # append the value into the appropriate list
                                 # based on column name k

print(columns['Precision'])
print(columns['Recall'])
precisions = columns['Precision']
recalls = columns['Recall']

precisions = list(map(float, precisions[:-1]))
recalls = list(map(float, recalls[:-1]))

# pyplot.plot([0, 1], [0, 1], linestyle='--')
print(len(precisions))
assert len(precisions) == len(recalls)
for i in range(0, len(precisions)-1):
    x1, y1 = [precisions[i], precisions[i+1]], [recalls[i], recalls[i+1]]
    pyplot.plot(x1, y1, '-')

pyplot.axis('equal')
pyplot.xlim([0.0, 1.0])
pyplot.ylim([0.0, 1.0])
pyplot.xlabel('Precision')
pyplot.ylabel('Recall')
pyplot.legend(loc="lower right")
pyplot.title('Precision Recall Curve')
pyplot.savefig('Precision-Recall-Curve.png')
pyplot.show()
