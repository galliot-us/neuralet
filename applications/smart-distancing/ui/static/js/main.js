function makeplot(objects_log_path) {
    Plotly.d3.csv(objects_log_path, function (data) {
        processData(data)
    });
}

function processData(allRows) {
    var x1 = [], y1 = [], y2 = [], env_score = [];
    allRows.forEach(function (element) {
        x1.push(element['Timestamp']);
        y1.push(element['DetectedObjects']);
        y2.push(element['ViolatingObjects']);
        env_score.push((element['EnvironmentScore']))
    });
    makePlotly(x1, y1, y2)
    makePlotlyEnvScore(x1, env_score)
}
function makePlotlyEnvScore(x, y) {
	var score = {
        x: x,
        y: y,
        fill: 'tozeroy',
        type: 'scatter',
        name: 'Environment Score'
    };
        Plotly.newPlot('envScore', [score],
        {title: 'Plotting log data Physical Distancing - Environment Score'});
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
