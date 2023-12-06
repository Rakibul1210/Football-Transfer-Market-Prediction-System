import React from "react"
import Button from 'react-bootstrap/Button';
import backgroundImage from './two.jpg'
import { useNavigate, useLocation } from 'react-router-dom';

const TitlePage = () => {
    const navigate = useNavigate();

    const styles = {
        backgroundImage: `url(${backgroundImage})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        minHeight: '100vh', // Ensure the image covers the full viewport height
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
      };

    const addData = (data) => {
        navigate('/FileReaderComponent');
    }

    return(
     <>
    <div style={{...styles}}>    
    <div style={{ marginTop:'270px',marginLeft: '150px',display:'block'  }}>   
     <Button variant="primary" style={{marginLeft: '360px',height: '50px', width: '150px',backgroundColor:'maroon',display:'block'}} className='col-lg-2' onClick={addData} type="submit">  Get Started </Button> 
     <p style={{ marginTop:'20px',fontSize:'40px',fontWeight:'bold', fontFamily:'cursive', color: 'white',display:'block' }}>Welcome to Football Transfer Market Predictor</p> 

     </div>
     </div>
    </>
    );
}
export default TitlePage;