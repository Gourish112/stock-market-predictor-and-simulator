import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";

const Home = () => {
  const [typedText, setTypedText] = useState("");
  const fullText = "The Ultimate Stock Predictor & Simulator";

  useEffect(() => {
    let i = 0;
    const typingEffect = setInterval(() => {
      setTypedText(fullText.slice(0, i));
      i++;
      if (i > fullText.length) clearInterval(typingEffect);
    }, 100);
    return () => clearInterval(typingEffect);
  }, []);

  return (
    <div style={styles.container}>
      {/* Welcome Heading with Neon Glow */}
      <h1 style={styles.heading}>
        <span style={styles.glowText}>Welcome to</span>
        <br />
        <span style={styles.highlight}>TradeLens</span>
      </h1>

      {/* Typing Effect Text */}
      <p style={styles.typingText}>{typedText}</p>

      {/* Features Section */}
      <div style={styles.featuresContainer}>
        <div style={styles.card}>
          <h2 style={styles.cardTitle}>📈 AI Stock Predictor</h2>
          <p style={styles.cardText}>
            Get real-time stock predictions using AI.
          </p>
        </div>

        <div style={styles.card}>
          <h2 style={styles.cardTitle}>💹 Stock Simulator</h2>
          <p style={styles.cardText}>
            Simulate trading with real-time data.
          </p>
        </div>
      </div>

      {/* CTA Buttons */}
      <div style={styles.buttonContainer}>
        <Link to="/predictor" style={styles.ctaButton}>
          Try Predictor
        </Link>
        <Link to="/simulator" style={styles.ctaButton}>
          Start Simulation
        </Link>
      </div>
    </div>
  );
};

// Styles with Black Background & Animations
const styles = {
  container: {
    textAlign: "center",
    color: "#fff",
    backgroundColor: "black",
    minHeight: "100vh",
    padding: "50px 20px",
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    position: "relative",
    overflow: "hidden",
  },
  heading: {
    fontSize: "4rem",
    fontWeight: "bold",
    marginBottom: "10px",
    animation: "fadeIn 1.5s ease-in-out",
  },
  glowText: {
    color: "white",
    textShadow: "0px 0px 10px rgba(255,255,255,0.8)",
    animation: "pulse 2s infinite alternate",
  },
  highlight: {
    color: "red",
    fontWeight: "bold",
    textShadow: "0px 0px 15px rgba(255,0,0,0.8)",
  },
  typingText: {
    fontSize: "1.6rem",
    marginBottom: "30px",
    opacity: 0.9,
    fontFamily: "monospace",
    borderRight: "3px solid #fff",
    whiteSpace: "nowrap",
    overflow: "hidden",
    width: "fit-content",
    animation: "blink 0.8s step-end infinite",
  },
  featuresContainer: {
    display: "flex",
    justifyContent: "center",
    gap: "30px",
    flexWrap: "wrap",
    animation: "slideUp 2s ease-in-out",
  },
  card: {
    background: "rgba(255, 255, 255, 0.1)",
    backdropFilter: "blur(10px)",
    padding: "20px",
    borderRadius: "10px",
    width: "300px",
    boxShadow: "0px 5px 15px rgba(0,0,0,0.5)",
    textAlign: "center",
    transition: "all 0.3s ease-in-out",
    cursor: "pointer",
    border: "1px solid rgba(255, 255, 255, 0.3)",
    animation: "fadeIn 2.5s ease-in-out",
  },
  cardTitle: {
    fontSize: "1.8rem",
    marginBottom: "10px",
    fontWeight: "bold",
  },
  cardText: {
    fontSize: "1.2rem",
    opacity: 0.9,
  },
  buttonContainer: {
    display: "flex",
    gap: "20px",
    marginTop: "30px",
  },
  ctaButton: {
    backgroundColor: "white",
    color: "black",
    padding: "15px 25px",
    fontSize: "18px",
    fontWeight: "bold",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
    transition: "transform 0.3s ease-in-out",
    textDecoration: "none",
  },
};

// Adding CSS Animations using JavaScript
const stylesTag = document.createElement("style");
stylesTag.innerHTML = `
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(-20px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes slideUp {
  from { opacity: 0; transform: translateY(50px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes pulse {
  0% { text-shadow: 0px 0px 5px rgba(255,255,255,0.8); }
  100% { text-shadow: 0px 0px 15px rgba(255,255,255,1); }
}

@keyframes blink {
  50% { border-color: transparent; }
}
`;
document.head.appendChild(stylesTag);

export default Home;
