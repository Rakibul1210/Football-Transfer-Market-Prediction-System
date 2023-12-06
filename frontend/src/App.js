import {Route, Routes} from 'react-router-dom' 
import TitlePage from './TitlePage';
import 'bootstrap/dist/css/bootstrap.min.css';
import FileReaderComponent from './FileReaderComponent';
import DisplayFileContent from './DisplayFileContent'; 
import PricePrediction from './PricePrediction';

function App() {
  return (
    <>
    <Routes>
      <Route path="/" element={<TitlePage/>}/>
      <Route path="/FileReaderComponent" element={<FileReaderComponent/>}/>
      <Route path="/DisplayFileContent" element={<DisplayFileContent/>}/>
      <Route path="/PricePrediction" element={<PricePrediction/>}/>



    </Routes>
    </>

  );
}

export default App;
