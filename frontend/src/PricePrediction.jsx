import React, { useState } from 'react';
import Button from 'react-bootstrap/Button';
import { useNavigate, useLocation } from 'react-router-dom';
import NavigationBar from './NavigationBar';
import axios from 'axios';
import backgroundImage from './three.jpg'

const PricePrediction = () => {
  const styles = {
    backgroundImage: `url(${backgroundImage})`,
    backgroundSize: 'cover',
    backgroundPosition: 'center',
    minHeight: '100vh', // Ensure the image covers the full viewport height
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
  };

    const location = useLocation();
    const playerData = location.state.fileContent;
    console.log('data: ',playerData)
    const predictedPrice = location.state.predictedPrice;
  return (
    <>
    <NavigationBar/>
    <div style={{...styles}}>    

    <div style={{ marginRight: '500px', marginLeft: '500px',marginBottom:'200px',alignItems: 'center', backgroundColor:'white', borderWidth:"10px", borderHeight:"10px",borderRadius:"10px", opacity:"80%"}}>
      <h2 style={{ fontFamily: 'monospace', color: 'maroon', marginBottom: '30px' }}>Price Prediction</h2>
        <div style={{ fontWeight: 'bold', fontSize: '40px',marginLeft:'75px' }}>{playerData["Name"]}</div>
        <div style={{ fontWeight: 'bold', fontFamily: 'cursive',color: 'maroon',fontSize: '30px',marginLeft:'75px'  }}>{predictedPrice}</div>
    </div>
    </div>
    </>
  );
};

export default PricePrediction;
