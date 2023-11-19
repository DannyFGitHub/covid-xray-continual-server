import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

import os

cred = credentials.Certificate("./covid-19-pneumocheck-firebase-adminsdk-4hjhg-65d0e6f2d1.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def get_image_from_firestore(submission_id):
    return db.collection('submissions').document(submission_id).get().to_dict()['imageUrl']

def set_prediction_in_firestore(submission_id, prediction, probabilities_dict):
    print("set_prediction_in_firestore", submission_id, prediction, probabilities_dict)
    # If document exists update
    document = db.collection('submissions').document(submission_id)
    if(document.get().exists):
        db.collection('submissions').document(submission_id).update({'modelName': "continual-covid-xray-cloud", 'probabilities': probabilities_dict})
    else:
        print("Document doesn't exist to post prediction to.")

def set_learnt_in_firestore(submission_id):
    print("set_learnt_in_firestore", submission_id)
    # If document exists update
    document = db.collection('submissions').document(submission_id)
    if(document.get().exists):
        db.collection('submissions').document(submission_id).update({'learntAt': firestore.SERVER_TIMESTAMP})
    else:
        print("Document doesn't exist to post learntAt to.")

def set_prediction_in_test_batch_firestore(submission_id, prediction, probabilities_dict, image_url, isLast):
    # If test_batch collection doesn't exist create it
    test_batch_collection = db.collection('test_batch')
    document = test_batch_collection.document(submission_id)
    if(document.get().exists):
        body = {'predictions': firestore.ArrayUnion([{'modelName': "continual-covid-xray-cloud", 'probabilities': probabilities_dict, 'prediction': prediction, 'imageUrl': image_url}])}
        if isLast == True:
            body['endedAt'] = firestore.SERVER_TIMESTAMP
        # Add to array in submission_id
        db.collection('test_batch').document(submission_id).update(body)
    else:
        body = {'startedAt': firestore.SERVER_TIMESTAMP, 'predictions': [{'modelName': "continual-covid-xray-cloud", 'probabilities': probabilities_dict, 'prediction': prediction, 'imageUrl': image_url}]}
        db.collection('test_batch').document(submission_id).set(body)
    


# This is only to download the mobile tests to csv
def get_local_test_batch_from_firestore_and_save_to_csv():
    # if it doesn't exist create a local folder to store the csvs
    if not os.path.exists('./local_test_batch'):
        os.makedirs('./local_test_batch')

    local_test_batch_collection = db.collection('local_test_results')
    docs = local_test_batch_collection.stream()
    for doc in docs:
        submission_id = doc.id
        # create a folder for the submission id
        if not os.path.exists('./local_test_batch/' + submission_id):
            os.makedirs('./local_test_batch/' + submission_id)
        
        # Get deviceModel
        deviceModel = doc.to_dict()['deviceModel']
        # Get Device Info
        deviceInfo = doc.to_dict()['deviceInfo']

        # Get inferenceTimeResults
        inferenceTimeResultsCSV = doc.to_dict()['inferenceTimeResultsCSV']
        # Get inferenceTimeTestsCSV
        inferenceTimeTestsCSV = doc.to_dict()['inferenceTimeTestsCSV']
        # Get loadingTimeResultsCSV
        loadingTimeResultsCSV = doc.to_dict()['loadingTimeResultsCSV']
        # Get loadingTimeTestsCSV
        loadingTimeTestsCSV = doc.to_dict()['loadingTimeTestsCSV']

        # Save inferenceTimeResultsCSV
        with open('./local_test_batch/' + submission_id + '/inferenceTimeResultsCSV-' + deviceModel + '.csv', 'w') as f:
            f.write(inferenceTimeResultsCSV)
        # Save inferenceTimeTestsCSV
        with open('./local_test_batch/' + submission_id + '/inferenceTimeTestsCSV-' + deviceModel + '.csv', 'w') as f:
            f.write(inferenceTimeTestsCSV)
        # Save loadingTimeResultsCSV
        with open('./local_test_batch/' + submission_id + '/loadingTimeResultsCSV-' + deviceModel + '.csv', 'w') as f:
            f.write(loadingTimeResultsCSV)
        # Save loadingTimeTestsCSV
        with open('./local_test_batch/' + submission_id + '/loadingTimeTestsCSV-' + deviceModel + '.csv', 'w') as f:
            f.write(loadingTimeTestsCSV)
        # Save device info
        with open('./local_test_batch/' + submission_id + '/deviceInfo-' + deviceModel + '.csv', 'w') as f:
            f.write(deviceInfo)

def get_submissions_from_firestore():
    # Filter by learntAt == null or prediction == UNCONFIRMED
    submissions_collection = db.collection('submissions')
    docs = submissions_collection.stream()
    submissions = []
    for doc in docs:
        submission = doc.to_dict()
        submission['submission_id'] = doc.id
        submissions.append(submission)
    return submissions

def get_submissions_from_firestore_with_date(since_date_updatedat_or_createdat):
    # Filter by learntAt == null or prediction == UNCONFIRMED
    submissions_collection = db.collection('submissions')
    # Get docs that were updated or created after since_date_updatedat_or_createdat
    docs = submissions_collection.where('submissionDate', '>=', since_date_updatedat_or_createdat.timestamp() * 1000).stream()
    docs_count = 0
    submissions = []
    for doc in docs:
        docs_count += 1
        submission = doc.to_dict()
        submission['submission_id'] = doc.id
        submissions.append(submission)
    if(docs_count > 0):
        print("docs_count", docs_count)
    return submissions


if __name__ == "__main__":
    get_local_test_batch_from_firestore_and_save_to_csv()