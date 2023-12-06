import React from 'react';
import { Navbar, Nav } from 'react-bootstrap';

const NavigationBar = () => {
  return (
<Navbar style={{ backgroundColor: '#E3BC9A', height:'70px' }} expand='lg'>
      <Navbar.Brand href="#home" style={{marginLeft:'40px', color: 'maroon',fontSize: '25px', fontWeight: 'bold', fontFamily: 'cursive'}}>Football Transfer Market Predictor</Navbar.Brand>
      <Navbar.Toggle aria-controls="basic-navbar-nav" />
      {/* <Navbar.Collapse id="basic-navbar-nav">
        <Nav className="ml-auto">
          <Nav.Link href="#home">Home</Nav.Link>
          <Nav.Link href="#about">About</Nav.Link>
          <Nav.Link href="#services">Services</Nav.Link>
          <Nav.Link href="#contact">Contact</Nav.Link>
        </Nav>
      </Navbar.Collapse> */}
    </Navbar>
  );
};

export default NavigationBar;
