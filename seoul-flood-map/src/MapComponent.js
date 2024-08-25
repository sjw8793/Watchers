import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';

const MapUpdater = ({ center, zoom }) => {
  const map = useMap();
  useEffect(() => {
    map.setView(center, zoom);
  }, [center, zoom, map]);
  return null;
};

const MapComponent = ({ selectedPosition, selectedGu, selectedDong }) => {
  const [floodData, setFloodData] = useState([]);
  const defaultZoom = 20;

  useEffect(() => {
    if (selectedGu && selectedDong) {
      const requestBody = {
        gu_name: selectedGu,
        dong_name: selectedDong,
      };

      console.log('Sending request to server:', requestBody);

      fetch('http://15.165.88.208:8000/predict/', { // API 엔드포인트 수정 필요
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      })
      .then(response => {
        console.log('Response status:', response.status);
        if (!response.ok) {
          throw new Error(`Server error: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        console.log('Received data from server:', data);
        if (Array.isArray(data)) {
          setFloodData(data);
        } else {
          console.error('Unexpected data format:', data);
          setFloodData([]);
        }
      })
      .catch(error => {
        console.error('Error fetching flood data:', error);
        setFloodData([]); // 데이터가 없거나 오류가 발생한 경우 빈 배열로 설정
      });
    }
  }, [selectedGu, selectedDong]);

  const getMarkerColor = (percent) => {
    const hue = 60 + (percent * 180); // percent가 0일 때 60 (노란색), percent가 1일 때 240 (파란색)
    return `hsl(${hue}, 100%, 50%)`;
  };

  return (
    <MapContainer center={selectedPosition} zoom={defaultZoom} style={{ height: '100vh', width: '100%' }}>
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      />
      {floodData.map((location, index) => (
        <CircleMarker
          key={index}
          center={[location.latitude, location.longitude]}
          radius={5} // 점의 크기
          color={getMarkerColor(location['flood-percent'])}
          fillColor={getMarkerColor(location['flood-percent'])}
          fillOpacity={0.8}
        >
          <Popup>
            {`Flood Probability: ${(location['flood-percent'] * 100).toFixed(2)}%`}
          </Popup>
        </CircleMarker>
      ))}
      <MapUpdater center={selectedPosition} zoom={defaultZoom} />
    </MapContainer>
  );
};

export default MapComponent;
