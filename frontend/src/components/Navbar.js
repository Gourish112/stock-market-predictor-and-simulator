import React from "react";
import { Link } from "react-router-dom";

const Navbar = () => {
  const navStyle = {
    backgroundColor: "black",
    color: "white",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "15px 5%",
    position: "fixed",
    width: "90%",
    top: 0,
    zIndex: 1000,
  };

  const logoStyle = {
    fontSize: "24px",
    fontWeight: "bold",
    cursor: "pointer",
    transition: "transform 0.3s ease-in-out",
  };

  const navLinksStyle = {
    display: "flex",
    gap: "20px",
    listStyle: "none",
    margin: 0,
    padding: 0,
  };

  const linkStyle = {
    color: "white",
    textDecoration: "none",
    fontSize: "18px",
    fontWeight: "500",
    transition: "color 0.3s ease-in-out",
  };

  const buttonStyle = {
    backgroundColor: "white",
    color: "black",
    padding: "10px 20px",
    fontSize: "16px",
    fontWeight: "bold",
    border: "none",
    cursor: "pointer",
    transition: "background 0.3s ease-in-out",
  };

  return (
    <nav style={navStyle}>
      <div style={logoStyle}>
        <Link to="/" style={{ color: "white", textDecoration: "none" }}>
          TradeLens
        </Link>
      </div>
      <ul style={navLinksStyle}>
        <li>
          <Link to="/" style={linkStyle}>
            Home
          </Link>
        </li>
        <li>
          <Link to="/predictor" style={linkStyle}>
            Predictor
          </Link>
        </li>
        <li>
          <Link to="/simulator" style={linkStyle}>
            Simulator
          </Link>
        </li>
      </ul>
      <Link to="/predictor">
        <button style={buttonStyle}>Get Started</button>
      </Link>
    </nav>
  );
};

export default Navbar;
