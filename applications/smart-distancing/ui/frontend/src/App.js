import React from 'react';
import {Container, AppBar, Tabs, Tab, Typography, Toolbar} from '@material-ui/core';
import {makeStyles} from '@material-ui/core/styles';
import {Switch, Route, Link, Redirect} from "react-router-dom";
import Live from "./Live";
import Offline from "./Offline";
import Reports from "./Reports";
import Settings from "./Settings";

const useStyles = makeStyles((theme) => ({
    root: {
        flexGrow: 1,
        backgroundColor: theme.palette.background.paper,
    },
    grow: {
        flexGrow: 1,
    },
    container: {
        paddingTop: theme.spacing(3),
        paddingBottom: theme.spacing(3),
    },
    content: {
        flexGrow: 1,
        height: '100vh',
        overflow: 'auto',
    },
}));

const tabProps = (to) => ({value: to, to: to, component: Link});

export default function App() {
    const classes = useStyles();

    return (
        <div className={classes.root}>
            <Route
                path="/"
                render={({location}) => (
                    <React.Fragment>
                        <AppBar position="static" title="hellooo">
                            <Container maxWidth="lg">
                                <Toolbar>
                                    <Tabs value={location.pathname}>
                                        <Tab label="Live" {...tabProps('/live')}/>
                                        {process.env.NODE_ENV === 'development' ?
                                            <Tab label="Offline" {...tabProps('/offline')}/> : null}
                                        {process.env.NODE_ENV === 'development' ?
                                            <Tab label="Reports" {...tabProps('/reports')}/> : null}
                                        {process.env.NODE_ENV === 'development' ?
                                            <Tab label="Settings" {...tabProps('/settings')}/> : null}
                                    </Tabs>
                                    <div className={classes.grow}/>
                                    <Typography variant="h6">
                                        {document.title}
                                    </Typography>
                                </Toolbar>
                            </Container>
                        </AppBar>
                        <main className={classes.content}>
                            <Container maxWidth="lg" className={classes.container}>
                                <Switch>
                                    <Route exact path="/">
                                        <Redirect to="/live"/>
                                    </Route>
                                    <Route path="/live">
                                        <Live/>
                                    </Route>
                                    <Route path="/offline">
                                        <Offline/>
                                    </Route>
                                    <Route path="/reports">
                                        <Reports/>
                                    </Route>
                                    <Route path="/settings">
                                        <Settings/>
                                    </Route>
                                </Switch>
                            </Container>
                        </main>
                    </React.Fragment>
                )}
            />
        </div>
    );
}
