// MapComponent 컴포넌트 : 선택된 위치에 따라 지도를 업데이트하고, 홍수 데이터를 표시 
import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';

// 지도 중심과 줌을 업데이트하는 컴포넌트 
const MapUpdater = ({ center, zoom }) => {
  const map = useMap();
  useEffect(() => {
    map.setView(center, zoom); // 지도 중심과 줌 레벨 업데이트 
  }, [center, zoom, map]);
  return null;
};

// 지도와 홍수 데이터 표시 컴포넌트 
const MapComponent = ({ selectedPosition, selectedGu, selectedDong }) => {
  const [floodData, setFloodData] = useState([]); // 서버에서 받아올 홍수 데이터 상태 
  const defaultZoom = 20; // 지도 기본 줌 레벨 

  useEffect(() => {
    if (selectedGu && selectedDong) {
      const requestBody = { //API 요청문 
        gu_name: selectedGu,
        dong_name: selectedDong,
      };

      console.log('Sending request to server:', requestBody); // 서버에 요청 보내기 

      fetch('http://15.165.88.208:8000/predict/', { // API 엔드포인트 - 구와 동에 해당하는 침수 확률 리턴하는 API
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody), // 요청 본문에 선택된 구와 동 정보 포함되어 있음 
      })
      .then(response => {
        console.log('Response status:', response.status); // 응답 상태 코드 로그 출력 
        if (!response.ok) {
          throw new Error(`Server error: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        console.log('Received data from server:', data); // 서버에서 받은 데이터 로그 출력 
        if (Array.isArray(data)) {
          setFloodData(data); // 서버에서 받은 홍수 데이터를 상태에 저장 
        } else {
          console.error('Unexpected data format:', data); // 예상치 못한 데이터 형식 처리 
          setFloodData([]);
        }
      })
      .catch(error => {
        console.error('Error fetching flood data:', error); // 데이터 요청 중 오류 처리 
        setFloodData([]); // 데이터가 없거나 오류가 발생한 경우 빈 배열로 설정
      });
    }
  }, [selectedGu, selectedDong]);
  // 침수 확률에 따른 마커 색상 설정 함수 
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
      {/* 홍수 데이터를 기반으로 마커 생성 */} 
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
      {/* 지도 위치 업데이트 */} 
      <MapUpdater center={selectedPosition} zoom={defaultZoom} />
    </MapContainer>
  );
};

export default MapComponent;
