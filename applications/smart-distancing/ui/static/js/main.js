function makeplot(objects_log_path) {
    Plotly.d3.csv(objects_log_path, function (data) {
        processData(data)
    });
}

function processData(allRows) {
    var x1 = [], y1 = [], y2 = [];
    allRows.forEach(function (element) {
        x1.push(element['Timestamp']);
        y1.push(element['DetectedObjects']);
        y2.push(element['ViolatingObjects'])
    });
    makePlotly(x1, y1, y2)
}

function makePlotly(x1, y1, y2) {
    var detected_pedestrians = {
        x: x1,
        y: y1,
        fill: 'tozeroy',
        type: 'scatter',
        name: 'Detected Pedestrians'
    };

    var detected_violating_pedestrians = {
        x: x1,
        y: y2,
        fill: 'tozeroy',
        type: 'scatter',
        name: 'Detected Pedestrians In Unsafe Area'
    };
    var result = [detected_pedestrians, detected_violating_pedestrians];
    Plotly.newPlot('myDiv', result,
        {title: 'Plotting log data Physical Distancing'});
}

makeplot(objects_log_path);