import React from 'react';
import ReactDOM from 'react-dom';
import CssBaseline from '@material-ui/core/CssBaseline';
import {ThemeProvider} from '@material-ui/core/styles';
import App from './App';
import theme from './theme';
import {HashRouter} from "react-router-dom";

ReactDOM.render(
    <ThemeProvider theme={theme}>
        <CssBaseline/>
        <HashRouter>
            <App/>
        </HashRouter>
    </ThemeProvider>,
    document.getElementById('root')
);
