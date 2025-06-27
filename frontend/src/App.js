import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import Predictor from "./pages/Predictor";
import Simulator from "./pages/Simulator";

const App = () => {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/predictor" element={<Predictor />} />
        <Route path="/simulator" element={<Simulator />} />
      </Routes>
    </Router>
  );
};

export default App;
