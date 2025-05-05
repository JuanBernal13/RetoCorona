// src/components/RecommendedProductsSection.js
import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Form, Button, Spinner, Alert } from 'react-bootstrap';
import Slider from "react-slick";
import RecommendationProductCard from './RecommendationProductCard'; // B2B Card
import B2cRecommendationProductCard from './B2cRecommendationProductCard'; // B2C Card
// No longer need to import b2cRecommendationsMock here
// import { b2cRecommendationsMock } from '../data/mockData';

function RecommendedProductsSection({ userType }) {
  const [productName, setProductName] = useState('');
  const [recommendedProducts, setRecommendedProducts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasSearched, setHasSearched] = useState(false);

  // Reset state when userType changes
  useEffect(() => {
    setProductName('');
    setRecommendedProducts([]);
    setError(null);
    setHasSearched(false);
  }, [userType]);


  const fetchRecommendations = async (name) => {
    setLoading(true);
    setError(null);
    setRecommendedProducts([]);
    setHasSearched(true);

    // Determine the endpoint URL based on userType and the new server ports
    // B2C (person) requests go to port 5001
    // B2B (company) requests go to port 5000
    const baseUrl = userType === 'person' ? 'http://127.0.0.1:5001' : 'http://127.0.0.1:5000';
    const endpoint = '/recommend'; // Both servers use the same endpoint name

    console.log(`Fetching ${userType === 'person' ? 'B2C' : 'B2B'} recommendations for: ${name} from ${baseUrl}${endpoint}`);

    try {
      const response = await fetch(`${baseUrl}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        // Send product_name in the request body for both
        // user_type is determined by the port/endpoint now
        body: JSON.stringify({ product_name: name }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      // Both backends are expected to return { recommendations: [...] }
      setRecommendedProducts(data.recommendations);
    } catch (err) {
      setError(`Error al obtener recomendaciones: ${err.message}. Asegúrate de que los servicios de Python estén corriendo y el nombre del producto sea válido.`);
      console.error("Error fetching recommendations:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (event) => {
    setProductName(event.target.value);
  };

  const handleSearch = (event) => {
    event.preventDefault();
    if (productName.trim()) {
      fetchRecommendations(productName.trim());
    } else {
      setError("Por favor, ingresa el nombre de un producto.");
      setRecommendedProducts([]);
      setHasSearched(false);
    }
  };

  const carouselSettings = {
    dots: true,
    infinite: true,
    speed: 500,
    slidesToShow: userType === 'person' ? 5 : 4,
    slidesToScroll: 1,
    autoplay: true,
    autoplaySpeed: 4000,
    responsive: [
        {
            breakpoint: 1200,
            settings: {
                slidesToShow: userType === 'person' ? 4 : 3,
                slidesToScroll: 1,
                infinite: true,
                dots: true
            }
        },
        {
            breakpoint: 992,
            settings: {
                slidesToShow: userType === 'person' ? 3 : 2,
                slidesToScroll: 1,
                initialSlide: 2
            }
        },
        {
            breakpoint: 768,
            settings: {
                slidesToShow: userType === 'person' ? 2 : 1,
                slidesToScroll: 1
            }
        }
    ]
  };

  const searchPlaceholder = userType === 'person'
    ? 'Ingresa el nombre de un producto (ej: Grifería, Piso)'
    : 'Ingresa el nombre de un producto (ej: Producto_1)';

  const recommendationsTitle = userType === 'person'
    ? `Recomendaciones para ti (${productName}):`
    : `Recomendaciones para "${productName}":`;


  return (
    <section className="recommended-products-section text-center py-5">
      <Container>
        <h2>Encuentra Productos Similares o Relacionados</h2>

        <Row className="justify-content-center mb-4">
          <Col md={6}>
            <Form onSubmit={handleSearch}>
              <Form.Group className="d-flex">
                <Form.Control
                  type="text"
                  placeholder={searchPlaceholder}
                  value={productName}
                  onChange={handleInputChange}
                  className="me-2 rounded-0"
                />
                <Button variant="primary" type="submit" className="rounded-0">
                  Obtener Recomendaciones
                </Button>
              </Form.Group>
            </Form>
          </Col>
        </Row>

        {loading && (
          <div className="text-center">
            <Spinner animation="border" role="status">
              <span className="visually-hidden">Cargando...</span>
            </Spinner>
            <p>Buscando recomendaciones...</p>
          </div>
        )}

        {error && <Alert variant="danger">{error}</Alert>}

        {!loading && !error && hasSearched && recommendedProducts.length === 0 && (
          <p>No se encontraron recomendaciones para "{productName}". Intenta con otro producto o verifica el nombre.</p>
        )}

        {!loading && !error && recommendedProducts.length > 0 && (
           <>
             <h3>{recommendationsTitle.replace('{productName}', productName)}</h3>
             <div className="recommendations-carousel-container">
                 <Slider {...carouselSettings}>
                     {recommendedProducts.map((product, index) => (
                         <div key={index} className="px-2">
                             {userType === 'person' ? (
                                 // Pass product details expected by B2C card
                                 <B2cRecommendationProductCard product={product} />
                             ) : (
                                 // Pass product details expected by B2B card
                                 <RecommendationProductCard product={product} />
                             )}
                         </div>
                     ))}
                 </Slider>
             </div>
           </>
        )}
      </Container>
    </section>
  );
}

export default RecommendedProductsSection;