import React from "react";
import { Link } from "react-router-dom";

const Navbar = () => {
  return (
    <nav style={styles.nav}>
      <div style={styles.container}>
        <div style={styles.logo}>
          <Link to="/" style={styles.logoLink}>TradeLens</Link>
        </div>

        <ul style={styles.navLinks}>
          <li><Link to="/" style={styles.link}>Home</Link></li>
          <li><Link to="/predictor" style={styles.link}>Predictor</Link></li>
          <li><Link to="/simulator" style={styles.link}>Simulator</Link></li>
        </ul>
      </div>
    </nav>
  );
};

const styles = {
  nav: {
    backgroundColor: "black",
    position: "fixed",
    top: 0,
    left: 0,
    width: "100%",
    zIndex: 1000,
    borderBottom: "1px solid #222",
  },

  container: {
    maxWidth: "1200px",
    margin: "0 auto",
    padding: "15px 20px",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },

  logo: {
    fontSize: "22px",
    fontWeight: "bold",
  },

  logoLink: {
    textDecoration: "none",
    color: "white",
  },

  navLinks: {
    display: "flex",
    gap: "30px",
    listStyle: "none",
    margin: 0,
    padding: 0,
  },

  link: {
    color: "#ddd",
    textDecoration: "none",
    fontSize: "16px",
    transition: "0.3s",
  },
};

export default Navbar;