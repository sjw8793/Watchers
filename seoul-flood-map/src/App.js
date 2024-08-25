import React, { useState } from 'react';
import './App.css';
import 'leaflet/dist/leaflet.css';
import MapComponent from './MapComponent';
import { Dropdown, DropdownButton } from 'react-bootstrap';
import seoulData from './seoulData'; // 서울시 동 데이터가 포함된 JSON 파일
import guMapping from './gu.json'; // 구 한글-영문 매핑 JSON 파일
import dongMapping from './dong.json'; // 동 한글-영문 매핑 JSON 파일


//App 컴포넌트 : 서울시 지도와 구, 동 선택 기능을 제공합니다. 
function App() {
  const [selectedPosition, setSelectedPosition] = useState([37.5665, 126.9780]); // 초기 위치 서울 시청
  const [selectedGu, setSelectedGu] = useState(''); // 선택된 구 (영문)
  const [selectedGuKorean, setSelectedGuKorean] = useState(''); // 선택된 구 (한글)
  const [selectedDong, setSelectedDong] = useState(''); // 선택된 동 (영문)

  //사용자가 구를 선택했을 때 호출되는 함수 
  const handleGuSelect = (eventKey) => {
    const guEng = guMapping[eventKey]; // 한글 구 이름을 영문으로 변환
    setSelectedGu(guEng); //영문 구 이름을 state에 저장 
    setSelectedGuKorean(eventKey); // 한글 구 이름을 따로 저장
    setSelectedDong(''); // 구를 선택하면 동을 초기화
  };
  //사용자가 동을 선택했을 때 호출되는 함수 
  const handleDongSelect = (eventKey) => {
    const dongEng = dongMapping[eventKey]; // 한글 동 이름을 영문으로 변환
    const selectedLocation = seoulData.find(location => location.neighborhood === eventKey); //선택된 동에 해당하는 위치 정보 찾기 
    if (selectedLocation) {
      setSelectedDong(dongEng); //영문 동 이름을 상태에 저장 
      setSelectedPosition([selectedLocation.latitude, selectedLocation.longitude]); //선택된 동의 좌표로 지도의 위치를 이동 
    }
  };

  // 선택된 구(한글)에 따라 해당 구의 동 목록 필터링
  const filteredDongData = seoulData.filter(location => location.district === selectedGuKorean);

  return (
    <div className="App">
    {/*구 선택 드롭다운 메뉴*/} 
      <div style={{ position: 'absolute', top: 100, left: 10, zIndex: 1000 }}>
        <DropdownButton
          id="dropdown-basic-button"
          title="구 선택"
          onSelect={handleGuSelect}
          style={{ zIndex: 1001 }}
        >
          {/* 구 목록을 동적으로 생성 */} 
          {[...new Set(seoulData.map(location => location.district))].map(district => (
            <Dropdown.Item eventKey={district} key={district}>
              {district}
            </Dropdown.Item>
          ))}
        </DropdownButton>
        {/*동 선택 드롭다운 메뉴*/}
        <DropdownButton
          id="dropdown-dong-button"
          title="동 선택"
          onSelect={handleDongSelect}
          style={{ zIndex: 1001 }}
          disabled={!selectedGu}
        >
          {/*선택된 구에 해당하는 동 목록을 동적 생성 */} 
          {filteredDongData.map(location => (
            <Dropdown.Item eventKey={location.neighborhood} key={location.neighborhood}>
              {location.neighborhood}
            </Dropdown.Item>
          ))}
        </DropdownButton>
      </div>
      {/*지도 컴포넌트, 선택된 위치에 따라 지도 표시 */}
      <MapComponent selectedPosition={selectedPosition} selectedGu={selectedGu} selectedDong={selectedDong} />
        {/*현재 선택된 구와 동 정보를 화면에 표시 */} 
      <div style={{ position: 'absolute', bottom: 10, right: 10, backgroundColor: 'white', padding: '10px', borderRadius: '5px', zIndex: 1000 }}>
        <p>선택된 구: {selectedGuKorean}</p>
        <p>선택된 동: {selectedDong}</p>
      </div>
    </div>
  );
}

export default App;
