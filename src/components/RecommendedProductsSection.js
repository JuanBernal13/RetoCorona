import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Form, Button, Spinner, Alert } from 'react-bootstrap';
import Slider from "react-slick";
import RecommendationProductCard from './RecommendationProductCard';
import B2cRecommendationProductCard from './B2cRecommendationProductCard'; // Assuming you create a B2C card
import './RecommendedProductsSection.css';
import { b2cRecommendationsMock } from '../data/mockData'; // Import B2C mock data

function RecommendedProductsSection({ userType }) { // Receive userType as a prop
  const [productName, setProductName] = useState('');
  const [recommendedProducts, setRecommendedProducts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasSearched, setHasSearched] = useState(false);

  // Reset state when userType changes (e.g., after logout/new login)
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

    // In a real application, you might send a different request or include userType
    // in the request to the backend to get B2C or B2B specific recommendations.
    // For this example, we'll simulate B2C recommendations using mock data
    // and use the existing backend call for B2B.

    if (userType === 'person') {
      // Simulate fetching B2C recommendations (using mock data)
      // In a real scenario, this would be an API call to a B2C endpoint
      console.log(`Fetching B2C recommendations for: ${name}`);
      // Filter mock data if needed, or just use a subset
      const filteredMock = b2cRecommendationsMock.filter(p => p.nombre.toLowerCase().includes(name.toLowerCase()));
      setRecommendedProducts(filteredMock);
      setLoading(false);
    } else { // userType === 'company'
        console.log(`Fetching B2B recommendations for: ${name}`);
      try {
        const response = await fetch('http://127.0.0.1:5000/recommend', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ product_name: name }),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        setRecommendedProducts(data);
      } catch (err) {
        setError(`Error al obtener recomendaciones: ${err.message}. Asegúrate de que el servicio de Python esté corriendo y el nombre del producto sea válido.`);
        console.error("Error fetching recommendations:", err);
      } finally {
        setLoading(false);
      }
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

  // Configuration for the recommendations carousel
  const carouselSettings = {
    dots: true,
    infinite: true,
    speed: 500,
    slidesToShow: userType === 'person' ? 5 : 4, // Show more for B2C if cards are simpler
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
    : `Recomendaciones para "{productName}":`;


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
             {/* Mostrar recomendaciones en un carrusel */}
             <div className="recommendations-carousel-container">
                 <Slider {...carouselSettings}>
                     {recommendedProducts.map((product, index) => (
                         <div key={index} className="px-2">
                              {/* Render different card based on user type */}
                              {userType === 'person' ? (
                                <B2cRecommendationProductCard product={product} />
                              ) : (
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