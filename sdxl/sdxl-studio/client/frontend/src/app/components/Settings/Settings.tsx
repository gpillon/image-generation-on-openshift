import config from '@app/config';
import axios from 'axios';
import * as React from 'react';
import { useHistory, useLocation, useParams } from 'react-router-dom';
import Emitter from '../../utils/emitter';
import { Page, PageSection, Text, TextContent, TextVariants, Form, FormGroup, Button, TextInput, TextInputGroup, TextInputGroupMain, TextInputGroupUtilities, Flex, FlexItem, TabTitleIcon, Slider, SliderOnChangeEvent } from '@patternfly/react-core';
import { Tabs, Tab, TabTitleText } from '@patternfly/react-core';
import { EyeIcon } from '@patternfly/react-icons';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faBucket } from '@fortawesome/free-solid-svg-icons';
import SDLogo from '@app/assets/bgimages/stabilityai-logo.svg';

interface SettingsProps { }

class EndpointSettings {
    endpointUrl: string;
    endpointToken: string;

    constructor(endpointURL, endpointToken: string) {
        this.endpointUrl = endpointURL ?? '';
        this.endpointToken = endpointToken ?? '';
    }
}

const SettingsManagement: React.FunctionComponent<SettingsProps> = () => {
    const history = useHistory();
    const location = useLocation();
    const params = useParams();

    /* Tabs Management */

    const [activeTabKey, setActiveTabKey] = React.useState<string | number>(0);
    const handleTabClick = (
        event: React.MouseEvent<any> | React.KeyboardEvent | MouseEvent,
        tabIndex: string | number
    ) => {
        setActiveTabKey(tabIndex);
    };

    /* SDXL Settings Management */

    const [endpointSettings, setEndpointSettings] = React.useState<EndpointSettings>(new EndpointSettings('', ''));
    const [endpointSettingsChanged, setEndpointSettingsChanged] = React.useState<boolean>(false);

    const [showEndpointToken, setEndpointShowToken] = React.useState<boolean>(false);

    React.useEffect(() => {
        axios.get(`${config.backend_api_url}/settings/sdxl-endpoint`)
            .then((response) => {
                const { settings } = response.data;
                if (settings !== undefined) {
                    setEndpointSettings(new EndpointSettings(settings.endpointUrl, settings.endpointToken));
                }
            })
            .catch((error) => {
                console.error(error);
                Emitter.emit('error', 'Failed to fetch configuration settings.');
            })
    }, []);

    const handleEndpointChange = (value, field) => {
        setEndpointSettings(prevState => ({
            ...prevState,
            [field]: value,
        }));
        setEndpointSettingsChanged(true);
    };

    const handleSaveEndpointSettings = (event) => {
        event.preventDefault();
        axios.put(`${config.backend_api_url}/settings/sdxl-endpoint`, endpointSettings)
            .then((response) => {
                Emitter.emit('notification', { variant: 'success', title: '', description: 'Settings saved successfully!' });
                setEndpointSettingsChanged(false);
            })
            .catch((error) => {
                console.error(error);
                Emitter.emit('notification', { variant: 'warning', title: '', description: 'Saving failed with the error: ' + error });
            });
    };

    const handleTestEndpointConnection = (event) => {
        event.preventDefault();
        axios.post(`${config.backend_api_url}/settings/test-sdxl-endpoint`, endpointSettings)
            .then((response) => {
                Emitter.emit('notification', { variant: 'success', title: '', description: 'Connection successful!' });
            })
            .catch((error) => {
                console.error(error);
                Emitter.emit('notification', { variant: 'warning', title: '', description: 'Connection failed with the error: ' + error.response.data.message.error });
            });
    }

    /* Render */

    return (
        <Page className='buckets-list'>
            <PageSection>
                <TextContent>
                    <Text component={TextVariants.h1}>Settings</Text>
                </TextContent>
            </PageSection>
            <PageSection>
                <Tabs
                    activeKey={activeTabKey}
                    onSelect={handleTabClick}
                    aria-label="Settings Tabs"
                    role="region"
                >
                    <Tab eventKey={0}
                        title={
                            <>
                                <TabTitleIcon>
                                    <img className='tab-logo' src={SDLogo} alt="HuggingFace Logo" />
                                </TabTitleIcon>{' '}
                                <TabTitleText>SDXL Endpoint Settings</TabTitleText>{' '}
                            </>
                        }
                        aria-label="SDXL Endpoint settings">
                        <Form onSubmit={handleSaveEndpointSettings}
                            className='settings-form'>
                            <FormGroup label="URL" fieldId="url">
                                <TextInput
                                    className='form-settings-long'
                                    value={endpointSettings.endpointUrl}
                                    onChange={(_event, value) => handleEndpointChange(value, 'endpointUrl')}
                                    id="endpointUrl"
                                    name="endpointUrl"
                                />
                            </FormGroup>
                            <FormGroup label="Token" fieldId="token">
                                <TextInputGroup className='form-settings'>
                                    <TextInputGroupMain
                                        value={endpointSettings.endpointToken}
                                        onChange={(_event, value) => handleEndpointChange(value, 'endpointToken')}
                                        id="endpointToken"
                                        name="endpointToken"
                                        type={showEndpointToken ? 'text' : 'password'}
                                    />
                                    <TextInputGroupUtilities>
                                        <Button
                                            variant="plain"
                                            aria-label={showEndpointToken ? 'Hide token' : 'Show token'}
                                            onClick={() => setEndpointShowToken(!showEndpointToken)}
                                        >
                                            <EyeIcon />
                                        </Button>
                                    </TextInputGroupUtilities>
                                </TextInputGroup>
                            </FormGroup>
                            <Flex>
                                <FlexItem>
                                    <Button type="submit" className='form-settings-submit' isDisabled={!endpointSettingsChanged}>Save SDXL Endpoint Settings</Button>
                                </FlexItem>
                                <FlexItem>
                                    <Button className='form-settings-submit' onClick={handleTestEndpointConnection}>Test Connection</Button>
                                </FlexItem>
                            </Flex>
                        </Form>
                    </Tab>
                </Tabs>
            </PageSection>
        </Page>
    );
};

export default SettingsManagement;
