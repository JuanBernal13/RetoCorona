import React from 'react';
import { Card, Button } from 'react-bootstrap';
import './B2cRecommendationProductCard.css'; // Create this CSS file

function B2cRecommendationProductCard({ product }) {
  // This card will display B2C relevant information from the mock data
  // You'd replace the fields below with the actual B2C data structure
  return (
    <Card className="b2c-recommendation-product-card h-100 rounded-0 border">
      <div className="b2c-product-image-wrapper position-relative">
        {/* Assuming B2C mock data has an image field */}
        <Card.Img variant="top" src={product.imagen || 'https://via.placeholder.com/150'} alt={product.nombre} className="b2c-product-card-img rounded-0" />
        {/* Optional: Add overlay and buttons like in the main ProductCard */}
         <div className="b2c-product-overlay d-flex flex-column justify-content-center align-items-center">
            <Button variant="light" size="sm" className="mb-2 b2c-quick-view-button">Ver Producto</Button>
            <Button variant="primary" size="sm" className="b2c-add-to-cart-button">Añadir al Carrito</Button>
        </div>
      </div>
      <Card.Body className="d-flex flex-column justify-content-start p-2">
        <Card.Title className="b2c-product-name text-center mb-1">{product.nombre}</Card.Title>
        {/* Assuming B2C mock data has price and other relevant fields */}
        <Card.Text className="text-muted text-center mb-2"><small>{product.marca}</small></Card.Text>
        <Card.Text className="text-center"><strong>${product.precio || 'Precio no disponible'}</strong></Card.Text>
      </Card.Body>
    </Card>
  );
}

export default B2cRecommendationProductCard;