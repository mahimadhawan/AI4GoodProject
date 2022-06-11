import './App.css';

import {
    BrowserRouter as Router,
    Routes,
    Route,
    Navigate,
} from "react-router-dom";

// import Home component
import HomePage from "./components/HomePage";
// import Home component
import TestForm from "./components/TestForm";
// import About component
import Results from "./components/Results";


function App() {
    return (
        <>
            {/* This is the alias of BrowserRouter i.e. Router */}
            <Router>
                <Routes>
                    {/* This route is for home component 
          with exact path "/", in component props 
          we passes the imported component*/}
                    <Route exact path="/" element={HomePage} />

                    {/* This route is for home component 
          with exact path "/", in component props 
          we passes the imported component*/}
                    <Route exact path="/TestForm" element={TestForm} />

                    {/* This route is for about component 
          with exact path "/about", in component 
          props we passes the imported component*/}
                    <Route path="/Results" element={Results} />

                    {/* If any route mismatches the upper 
          route endpoints then, redirect triggers 
          and redirects app to home component with to="/" */}
                    <Navigate to="/" />
                </Routes>
            </Router>
        </>
    );
}

export default App;