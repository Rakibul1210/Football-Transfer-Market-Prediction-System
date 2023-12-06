import React, { useState } from 'react';
import NavigationBar from './NavigationBar';
import Button from 'react-bootstrap/Button';
import { useNavigate, useLocation } from 'react-router-dom';
import PricePrediction from './PricePrediction';
import axios from 'axios';

const DisplayFileContent = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const playerData = location.state.fileContent;
    console.log('content: ', location.state.fileContent)


    const keys = Object.keys(playerData);
    const rowCount = Math.ceil(keys.length / 3); // Rows per column
  
    // Splitting keys into three columns
    const column1 = keys.slice(0, rowCount);
    const column2 = keys.slice(rowCount, rowCount * 2);
    const column3 = keys.slice(rowCount * 2);

    const predictPrice = async () => {
        try {
            const response = await axios.post('http://10.100.201.211:5000/predict_transfer_fee', {playerData:playerData });
            console.log('test data:', response);
            const price = '$' + response.data.transfer_fee + 'M';
            navigate('/PricePrediction', { state: { fileContent: playerData, predictedPrice: price } }); 

          } catch (error) {
            console.error('Error testing data:', error);
          }

    };

  return (
    <>
    <NavigationBar/>
    <div style={{marginTop: '20px',marginLeft: '650px'}}>
      <h2 style={{fontWeight:'bold' ,fontSize: '30px',fontFamily: 'monospace', color: 'maroon', marginBottom: '60px'}}>Player Data</h2>
    </div>

    <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '20px',marginRight: '150px', marginLeft: '150px' }}>
      <div style={{ marginRight: '20px' }}>
        {column1.map((key) => (
          <div key={key} style={{ marginBottom: '10px' }}>
            <span style={{ marginRight: '10px', fontWeight: 'bold' }}>{key}: </span>
            <span>{playerData[key]}</span>
          </div>
        ))}
      </div>
      <div style={{ marginRight: '20px' }}>
        {column2.map((key) => (
          <div key={key} style={{ marginBottom: '10px' }}>
            <span style={{ marginRight: '10px', fontWeight: 'bold' }}>{key}: </span>
            <span>{playerData[key]}</span>
          </div>
        ))}
      </div>
      <div>
        {column3.map((key) => (
          <div key={key} style={{ marginBottom: '10px' }}>
            <span style={{ marginRight: '10px', fontWeight: 'bold' }}>{key}: </span>
            <span>{playerData[key]}</span>
          </div>
        ))}
      </div>
    </div>
      <Button variant="primary" style={{height: '40px',width:'150px', display: 'block',marginBottom: '30px',marginLeft: '660px', marginTop: '60px',backgroundColor:'maroon', color: 'white'}} className='col-lg-2' onClick={predictPrice} type="submit">  Predict price </Button>
    </>
  );
};

export default DisplayFileContent;