import React, { useState } from 'react';
import './App.css';
import 'leaflet/dist/leaflet.css';
import MapComponent from './MapComponent';
import { Dropdown, DropdownButton } from 'react-bootstrap';
import seoulData from './seoulData'; // 서울시 동 데이터가 포함된 JSON 파일
import guMapping from './gu.json'; // 구 한글-영문 매핑
import dongMapping from './dong.json'; // 동 한글-영문 매핑

function App() {
  const [selectedPosition, setSelectedPosition] = useState([37.5665, 126.9780]); // 초기 위치 서울 시청
  const [selectedGu, setSelectedGu] = useState(''); // 선택된 구 (영문)
  const [selectedGuKorean, setSelectedGuKorean] = useState(''); // 선택된 구 (한글)
  const [selectedDong, setSelectedDong] = useState(''); // 선택된 동 (영문)

  const handleGuSelect = (eventKey) => {
    const guEng = guMapping[eventKey]; // 한글 구 이름을 영문으로 변환
    setSelectedGu(guEng);
    setSelectedGuKorean(eventKey); // 한글 구 이름을 따로 저장
    setSelectedDong(''); // 구를 선택하면 동을 초기화
  };

  const handleDongSelect = (eventKey) => {
    const dongEng = dongMapping[eventKey]; // 한글 동 이름을 영문으로 변환
    const selectedLocation = seoulData.find(location => location.neighborhood === eventKey);
    if (selectedLocation) {
      setSelectedDong(dongEng);
      setSelectedPosition([selectedLocation.latitude, selectedLocation.longitude]);
    }
  };

  // 선택된 구(한글)에 따른 동 목록 필터링
  const filteredDongData = seoulData.filter(location => location.district === selectedGuKorean);

  return (
    <div className="App">
      <div style={{ position: 'absolute', top: 100, left: 10, zIndex: 1000 }}>
        <DropdownButton
          id="dropdown-basic-button"
          title="구 선택"
          onSelect={handleGuSelect}
          style={{ zIndex: 1001 }}
        >
          {[...new Set(seoulData.map(location => location.district))].map(district => (
            <Dropdown.Item eventKey={district} key={district}>
              {district}
            </Dropdown.Item>
          ))}
        </DropdownButton>

        <DropdownButton
          id="dropdown-dong-button"
          title="동 선택"
          onSelect={handleDongSelect}
          style={{ zIndex: 1001 }}
          disabled={!selectedGu}
        >
          {filteredDongData.map(location => (
            <Dropdown.Item eventKey={location.neighborhood} key={location.neighborhood}>
              {location.neighborhood}
            </Dropdown.Item>
          ))}
        </DropdownButton>
      </div>

      <MapComponent selectedPosition={selectedPosition} selectedGu={selectedGu} selectedDong={selectedDong} />

      <div style={{ position: 'absolute', bottom: 10, right: 10, backgroundColor: 'white', padding: '10px', borderRadius: '5px', zIndex: 1000 }}>
        <p>선택된 구: {selectedGuKorean}</p>
        <p>선택된 동: {selectedDong}</p>
      </div>
    </div>
  );
}

export default App;
