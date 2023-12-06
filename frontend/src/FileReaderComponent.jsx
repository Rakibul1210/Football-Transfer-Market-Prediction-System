import React, { useState } from 'react';
import Button from 'react-bootstrap/Button';
import NavigationBar from './NavigationBar';
import { useNavigate, useLocation } from 'react-router-dom';
import DisplayFileContent from './DisplayFileContent'; 
import axios from 'axios';
import backgroundImage from './one.jpg'

const FileReaderComponent = () => {
  const [fileContent, setFileContent] = useState('');
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

  const style1 = {
    
    justifyContent: 'center',
    alignItems: 'center',
};

// const trainData = async () => {
//   try {
//     const response = await axios.post('https://your-api-endpoint.com/train', { });
//     console.log('Training data:', response.data);

//   } catch (error) {
//     console.error('Error training data:', error)
//   }
// };

  const handleFileRead = async () => {
    try {
      const response = await fetch('/demo.txt'); // Adjust the path accordingly
      const text = await response.text();
      const parsedJSON = JSON.parse(text);

      setFileContent(parsedJSON);
      navigate('/DisplayFileContent', { state: { fileContent: parsedJSON } }); 
        } catch (error) {
      console.error('Error fetching the file:', error);
    }
  };

  return (
    <>
    <NavigationBar/>
    <div style={{...styles}}>    

        <div style={{marginLeft: '80px', ...style1}}>
        {/* <Button variant="primary" style={{height: '40px', marginBottom: '20px', display: 'block'}} className='col-lg-2' onClick={trainData} type="submit">  Train dataset </Button>   */}
         <Button variant="primary" style={{height: '50px', display: 'block',backgroundColor:'maroon'}} className='col-lg-12' onClick={handleFileRead} type="submit">  Read file for prediction </Button>
         </div>
       </div>
    </>

  );
};

export default FileReaderComponent;
